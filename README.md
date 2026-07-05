# Recursive Monte-Carlo Tree Search (RMCTS) — README

Maintained by 
Benjamin Howard (bhoward73@gmail.com, or bjhowa3@idaccr.org), and
Keith Frankston (k.frankston@fastmail.com, or k.frankston@idaccr.org)


This repository contains code to compare MCTS-UCB (AlphaZero style) and RMCTS (posterior / optimized policy) implementations and timings, plus tools to run inference with ONNX / TensorRT and PyCUDA. 

This code supports our ICML paper on RMCTS: 
[paper](https://openreview.net/forum?id=oo9523XUWI&referrer=%5BAuthor%20Console%5D%28%2Fgroup%3Fid%3DICML.cc%2F2026%2FConference%2FAuthors%23your-submissions%29)


The RMCTS code (the focus of this git repo) can be found in src/c/rmcts.c

Quick highlights
- Research code comparing two MCTS variants (timings, experiments).
- Tools for running TensorRT / ONNX inference and measuring latency.

Docker
- In this case, no manual installs (below) are required.
- Build a GPU-ready image from the repo root with `docker build -t rmcts .`
- It takes over eight minutes for this image to be created (w/ fast internet connection)
- To use a GPU, make sure you configure your docker to have access to your GPU(s).
- Then, `docker run -it --rm --gpus all rmcts` will put you in an interactive shell.
- From that point, you still need to build by running `./build_all_games.sh`.
- See below for commands to run the timing and quality tests from sections 6 and 7 of the paper.
- When you're finished, running `exit` will exit from the docker interactive shell environment.
- We thank Max-We for preparing this Dockerfile.

Supported / tested (approx.)
- Linux with NVIDIA GPU (CUDA) — TensorRT and PyCUDA require a compatible driver + CUDA toolkit.
- Python 3.8–3.12 (use a virtualenv or conda env).

Main dependencies
- Require Linux operating system (we used Ubuntu)
- gcc (we used version 13.3.0)
- numpy
- matplotlib (for plotting timing results)
- torch (PyTorch) — model training / exporting
- onnx — load/modify ONNX models
- onnxruntime (prefer `onnxruntime-gpu` if using CUDAExecutionProvider)
- tensorrt (NVIDIA TensorRT Python bindings) 
- pycuda — device allocations / async copies

Notes about GPU / vendor packages
- PyTorch: install a build that matches your CUDA (conda is easiest: see PyTorch website).
- TensorRT: typically installed via the system package (apt) or by following NVIDIA TensorRT installation instructions. The `tensorrt` Python package and matching runtime libraries are required for TensorRT paths.
- PyCUDA requires the CUDA toolkit and headers available at build time.
- onnxruntime: for GPU inference use `onnxruntime-gpu` that matches your CUDA/cuDNN (or use CPU-only `onnxruntime`).


AFTER INSTALLING THE ABOVE PACKAGES

To make a particular game, such as Othello:
```bash
make GAME=othello
```
this compiles relevant code and puts the results in ./build/othello

To make all three games, run:
```bash
./build_all_games.sh
```
It simply runs "make GAME=connect4; make GAME=dotbox; make GAME=othello;".

To perform your own timings (cf. Section 6 of the paper):
```bash
python othello_timings.py
python connect4_timings.py
python dotbox_timings.py
```

To run the Othello quality test pitting RMCTS vs MCTS-UCB:
```bash
python pit.py
```

To run the Othello strength saturation test RMCTS vs RMCTS (N sims to N/2 sims):
```bash
python saturationtest.py
```

To visualize the RMCTS search tree for Othello (requires a GPU):
```bash
python testRMCTS.py
python3 -m http.server 8000
```
Then open http://localhost:8000/tree_viewer.html and click "Load R_tree.json".
Cytoscape.js is loaded from CDN; no additional install is required.
See TREE_VIEWER.md for more details.

-----------------------------------------------------------------------------

Copyright (c) 2025, Institute for Defense Analyses, 730 Glebe Rd, Alexandria, VA 22305-3086; 703-845-2500

This material may be reproduced by or for the U.S. Government pursuant to all applicable FAR and DFARS clauses.
