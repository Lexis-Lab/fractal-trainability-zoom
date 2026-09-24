# %% [markdown]
# # Fractal boundary of neural-network trainability: zoom animation
# 
# Re-creates the experiment from Jascha Sohl-Dickstein, *The boundary of neural network trainability is fractal* (arXiv:2402.06184).
# 
# **Setup (as in the paper):** one-hidden-layer network (16 units, tanh), MSE loss, full-batch gradient descent. Weights, inputs and labels are all N(0,1), and the number of data points equals the number of parameters (272). Every pixel is one full training run with its own pair of learning rates (η0 for the hidden layer, η1 for the output layer). A run counts as *converged* if the mean of its last 20 losses is below the initial loss.
# 
# **Use:** Runtime → Change runtime type → GPU (T4 is fine), then Run all. Tweak the settings in the second cell.

# %%
# ============================ SETTINGS ============================
ACTIVATION   = "tanh"   # "tanh" or "relu"
NEURONS      = 16       # hidden width; data points = NEURONS*NEURONS + NEURONS (as in the paper)
STEPS        = 300      # gradient-descent steps per pixel (500 = sharper, slower)
DATA_SEED    = 0        # seed for weights, inputs, labels

RES          = 192      # frame resolution in pixels (384 = high quality, much slower)
FRAMES       = 120      # number of zoom frames
TOTAL_ZOOM   = 3e3      # total magnification. float32 supports ~3e3 at RES=192; use USE_FLOAT64 for deeper
START_REL_WIDTH = 0.2   # first frame spans +-20% of the centre learning rates

USE_FLOAT64  = False    # True allows zoom of 1e10+ but is much slower on Colab GPUs
CENTER       = None     # None = auto-pick a busy boundary point, or set (lr0, lr1) e.g. (1.3, 0.9)
CENTER_SEED  = 0        # change to get a different auto-picked location
RECENTER     = 0.3      # 0 = fixed centre, 0.3 = gently keep the camera on the boundary

SURVEY_RES   = 1024     # overview width in pixels (height is half of this)
SURVEY_LOG_RANGE0 = (0, 3)     # overview range of eta0 (log10): 1 .. 1000
SURVEY_LOG_RANGE1 = (0, 1.5)   # overview range of eta1 (log10): 1 .. ~30
# (for this network the boundary turns ragged/fractal around eta0 > 30, eta1 ~ 5)

FPS, UPSCALE, PINGPONG = 24, 2, True   # video: frames/s, pixel upscaling, zoom in then back out
OUT_MP4 = "fractal_zoom.mp4"

# %%
import time, base64
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import jax
if USE_FLOAT64:
    jax.config.update("jax_enable_x64", True)   # must happen before any JAX array is created
import jax.numpy as jnp
from jax import lax

DT = jnp.float64 if USE_FLOAT64 else jnp.float32
print("devices:", jax.devices(), "| dtype:", DT.__name__)

# ---------- data and initial weights (all N(0,1), fixed for every pixel) ----------
N_DATA = NEURONS * NEURONS + NEURONS
rng = np.random.default_rng(DATA_SEED)
X  = jnp.asarray(rng.standard_normal((N_DATA, NEURONS)), DT)
Y  = jnp.asarray(rng.standard_normal((N_DATA, 1)), DT)
W0 = jnp.asarray(rng.standard_normal((NEURONS, NEURONS)), DT)
W1 = jnp.asarray(rng.standard_normal((1, NEURONS)), DT)

if ACTIVATION == "tanh":
    act  = jnp.tanh
    dact = lambda h, pre: 1.0 - h**2
elif ACTIVATION == "relu":
    act  = lambda z: jnp.maximum(z, 0.0)
    dact = lambda h, pre: (pre > 0).astype(DT)
else:
    raise ValueError("ACTIVATION must be 'tanh' or 'relu'")

def forward(W0, W1):
    pre = X @ W0.T
    h = act(pre)
    return pre, h, h @ W1.T

L0 = float(jnp.mean((forward(W0, W1)[2] - Y) ** 2))   # initial loss, used to normalise the loss to 1
print(f"{N_DATA} data points, initial loss {L0:.3f}")

DIV = 1e8   # a run is declared diverged once its normalised loss exceeds this (or becomes NaN/inf)

def loss_and_grads(W0, W1):
    pre, h, yhat = forward(W0, W1)
    err = yhat - Y
    loss = jnp.mean(err ** 2) / L0
    g = 2.0 * err / (N_DATA * L0)          # dLoss/dyhat
    dW1 = g.T @ h                           # (1, n)
    dW0 = ((g @ W1) * dact(h, pre)).T @ X   # (n, n)
    return loss, dW0, dW1

def train(lr0, lr1):
    """Full-batch gradient descent for one (lr0, lr1) pair. Returns (score, step of divergence)."""
    def step(carry, t):
        w0, w1, active, tdiv = carry
        loss, d0, d1 = loss_and_grads(w0, w1)
        bad = (~jnp.isfinite(loss)) | (loss > DIV)
        tdiv = jnp.where(bad & (tdiv >= STEPS), t, tdiv)
        active = active & ~bad
        w0 = jnp.where(active, w0 - lr0 * d0, w0)   # freeze diverged runs (keeps NaNs out)
        w1 = jnp.where(active, w1 - lr1 * d1, w1)
        return (w0, w1, active, tdiv), jnp.where(active, loss, DIV)
    init = (W0, W1, jnp.array(True), jnp.array(STEPS, jnp.int32))
    (_, _, _, tdiv), losses = lax.scan(step, init, jnp.arange(STEPS, dtype=jnp.int32))
    return jnp.mean(losses[-20:]), tdiv          # score < 1  ->  converged

train_batch = jax.jit(jax.vmap(train))
CHUNK = 8192

def run_grid(lr0, lr1):
    """lr0, lr1: 2-D numpy arrays (same shape). One training run per element."""
    shp, a, b = lr0.shape, lr0.ravel(), lr1.ravel()
    n = a.size; pad = (-n) % CHUNK
    a = np.concatenate([a, np.full(pad, a[0])]); b = np.concatenate([b, np.full(pad, b[0])])
    S, T = [], []
    for i in range(0, a.size, CHUNK):
        s, t = train_batch(jnp.asarray(a[i:i+CHUNK], DT), jnp.asarray(b[i:i+CHUNK], DT))
        S.append(np.asarray(s)); T.append(np.asarray(t))
    return np.concatenate(S)[:n].reshape(shp), np.concatenate(T)[:n].reshape(shp)

# ---------- colouring: converged = dark teal (brighter = lower loss); diverged = fire (brighter = survived longer) ----------
CONV_CMAP = LinearSegmentedColormap.from_list("conv", ["#020617", "#0b3a5b", "#1e88a8", "#8fe3d8"])
DIV_CMAP  = plt.get_cmap("inferno")

def colorize(score, tdiv):
    conv = score < 1.0
    lc = np.clip(-np.log10(np.maximum(score, 1e-12)), 0, 6) / 6.0
    t = np.log10(tdiv + 1.0) / np.log10(STEPS + 1.0)
    rgb = np.where(conv[..., None], CONV_CMAP(lc)[..., :3], DIV_CMAP(np.clip(t, 0, 1))[..., :3])
    return (rgb * 255).astype(np.uint8)

def boundary_mask(conv):
    b = np.zeros_like(conv, dtype=bool)
    b[:-1, :] |= conv[:-1, :] != conv[1:, :]
    b[:, :-1] |= conv[:, :-1] != conv[:, 1:]
    return b

# %%
# ============ 1) OVERVIEW MAP (log-log) + automatic choice of a boundary point to zoom into ============
from scipy.ndimage import uniform_filter
t0 = time.time()
LR0, LR1 = np.meshgrid(np.logspace(*SURVEY_LOG_RANGE0, SURVEY_RES),
                       np.logspace(*SURVEY_LOG_RANGE1, SURVEY_RES // 2))   # rows = eta1, columns = eta0
score, tdiv = run_grid(LR0, LR1)
print(f"overview computed in {time.time()-t0:.1f}s")

conv = score < 1.0
bmask = boundary_mask(conv)

if CENTER is None:
    dens = uniform_filter(bmask.astype(float), size=15)          # how busy is the boundary here?
    dens[:8, :] = dens[-8:, :] = 0; dens[:, :8] = dens[:, -8:] = 0
    cand = np.flatnonzero(dens.ravel() >= np.quantile(dens[dens > 0], 0.99))
    pick = np.random.default_rng(CENTER_SEED).choice(cand)
    iy, ix = np.unravel_index(pick, dens.shape)
    CENTER = (float(LR0[iy, ix]), float(LR1[iy, ix]))
print(f"zoom centre: eta0={CENTER[0]:.5f}, eta1={CENTER[1]:.5f}")

fig, ax = plt.subplots(figsize=(9, 4.5))
ext = [*SURVEY_LOG_RANGE0, *SURVEY_LOG_RANGE1]
ax.imshow(colorize(score, tdiv), origin="lower", extent=ext, aspect="auto")
ax.plot(np.log10(CENTER[0]), np.log10(CENTER[1]), "w+", ms=14, mew=2)
ax.set_xlabel(r"$\log_{10}\,\eta_0$"); ax.set_ylabel(r"$\log_{10}\,\eta_1$")
ax.set_title("Trainability overview (white cross = zoom target)")
plt.show()

# %%
# ============ 2) ZOOM: one training run per pixel per frame ============
eps = np.finfo(np.float64 if USE_FLOAT64 else np.float32).eps
end_rel_spacing = START_REL_WIDTH / TOTAL_ZOOM * 2 / (RES - 1)
if end_rel_spacing < 4 * eps:
    print(f"WARNING: at the deepest frame neighbouring pixels differ by ~{end_rel_spacing:.1e} of the learning rate, "
          f"close to the {'float64' if USE_FLOAT64 else 'float32'} precision limit ({eps:.1e}). "
          f"Reduce TOTAL_ZOOM or RES, or set USE_FLOAT64 = True.")

c = np.array(CENTER, dtype=np.float64)
S = c.copy()                                       # axis scales, fixed
rhos = START_REL_WIDTH * TOTAL_ZOOM ** (-np.linspace(0, 1, FRAMES))
tt = np.linspace(-1, 1, RES)
mid = (RES - 1) / 2
frames, t0 = [], time.time()

for k, rho in enumerate(rhos):
    LR0, LR1 = np.meshgrid(c[0] + S[0] * rho * tt, c[1] + S[1] * rho * tt)
    score, tdiv = run_grid(LR0, LR1)
    frames.append(np.flipud(colorize(score, tdiv)))             # flip so eta1 increases upward

    b = boundary_mask(score < 1.0)                               # keep the camera on the boundary
    if RECENTER > 0 and b.any():
        iy, ix = np.nonzero(b)
        j = np.argmin((iy - mid) ** 2 + (ix - mid) ** 2)
        c[0] += RECENTER * (ix[j] - mid) / mid * S[0] * rho
        c[1] += RECENTER * (iy[j] - mid) / mid * S[1] * rho

    if k % 10 == 0 or k == FRAMES - 1:
        el = time.time() - t0
        print(f"frame {k+1}/{FRAMES}  zoom x{START_REL_WIDTH/rho:,.0f}  elapsed {el:.0f}s  eta {el/(k+1)*(FRAMES-k-1):.0f}s")

# preview of a few frames
idx = np.linspace(0, FRAMES - 1, 4).astype(int)
fig, axs = plt.subplots(1, 4, figsize=(16, 4.3))
for a_, i in zip(axs, idx):
    a_.imshow(frames[i]); a_.axis("off"); a_.set_title(f"zoom x{START_REL_WIDTH/rhos[i]:,.0f}")
plt.show()

# %%
# ============ 3) ENCODE VIDEO, PLAY, DOWNLOAD ============
import imageio.v2 as imageio
from PIL import Image

def up(f):
    return np.asarray(Image.fromarray(f).resize((RES * UPSCALE, RES * UPSCALE), Image.BICUBIC)) if UPSCALE != 1 else f

seq = frames + frames[-2:0:-1] if PINGPONG else frames    # zoom in, then back out -> seamless loop
with imageio.get_writer(OUT_MP4, fps=FPS, codec="libx264", quality=8, macro_block_size=1) as w:
    for f in seq:
        w.append_data(up(f))
Image.fromarray(up(frames[-1])).save("fractal_deepest_frame.png")
print("saved", OUT_MP4, "and fractal_deepest_frame.png")

from IPython.display import HTML, display
b64 = base64.b64encode(open(OUT_MP4, "rb").read()).decode()
display(HTML(f'<video width="520" controls loop autoplay muted><source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>'))

try:
    from google.colab import files
    files.download(OUT_MP4); files.download("fractal_deepest_frame.png")
except ImportError:
    pass