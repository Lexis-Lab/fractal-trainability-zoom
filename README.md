# fractal-trainability-zoom
Zoom animation of the fractal boundary of neural network trainability
# Fractal boundary of neural network trainability: zoom animation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lexis-Lab/fractal-trainability-zoom/blob/main/fractal_trainability_zoom.ipynb)

A Google Colab notebook that generates a zoom animation of the boundary between learning rates where neural network training converges and where it diverges. That boundary is fractal.

Based on "The boundary of neural network trainability is fractal" by Jascha Sohl-Dickstein ([arXiv:2402.06184](https://arxiv.org/abs/2402.06184)). His original code and images: https://github.com/Sohl-Dickstein/fractal

## What it does

- Trains a one-hidden-layer network (16 tanh units, MSE loss, full-batch gradient descent) with weights, inputs and labels drawn from N(0,1) and 272 data points.
- Each pixel is one full training run with its own pair of learning rates (η0 for the hidden layer, η1 for the output layer).
- A run counts as converged if the mean of its last 20 losses is below the initial loss.
- Renders an overview map, then zooms into the boundary frame by frame and exports an MP4.

Colours: converged runs are teal (brighter means lower final loss), diverged runs are fire-coloured (brighter means the run survived longer before blowing up).

## How to run

1. Click the **Open in Colab** badge above.
2. Choose **Runtime → Change runtime type → GPU**.
3. Choose **Runtime → Run all**. The video is played in the notebook and downloaded.

## Settings

All settings are in the second cell of the notebook.

| Setting | Meaning |
|---|---|
| `RES`, `FRAMES` | Frame resolution and number of zoom frames |
| `STEPS` | Gradient-descent steps per pixel (more gives sharper edges but is slower) |
| `TOTAL_ZOOM` | Total magnification (about 3e3 is the safe limit in float32) |
| `USE_FLOAT64` | Allows much deeper zooms, but is slower on Colab GPUs |
| `CENTER` | `None` picks a boundary point automatically, or set `(eta0, eta1)` yourself |
| `CENTER_SEED` | Change it to get a different automatically picked location |

For this network the boundary is roughly flat near η1 ≈ 6–7 and becomes ragged for η0 above about 30. If the automatic zoom looks like noise, try setting `CENTER` by hand, for example `(40, 6.47)` with `START_REL_WIDTH = 0.1`.

<!-- After uploading your video or GIF to this repo, delete this line's comment markers and fix the filename:
![zoom](fractal_zoom.gif)
-->

## Files

- `fractal_trainability_zoom.ipynb`: the Colab notebook
- `fractal_trainability_zoom.py`: the same code as a plain script

## Credit

The experiment and idea belong to Jascha Sohl-Dickstein. This notebook is an independent re-implementation of his setup.
