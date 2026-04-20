# Dataset Instructions

This repository does **not** ship raw dataset binaries to keep the project lightweight and GitHub-friendly.

## What is included
- dataset generation scripts (`data_gen.py`, `data_extend.py`)
- metadata (`Meta.py`)

## What you need to provide
Place your `.npy` (or other required source files) under dataset subfolders before training. Example:

- `dataset/Pendulum/Pendulum.npy`
- `dataset/Lorenz/Lorenz.npy`
- `dataset/KS/KS_8pi.npy`

Then run experiments with `--root_path` and `--data_path`