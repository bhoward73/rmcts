# RMCTS — README

Maintained by 
Benjamin Howard (bhoward73@gmail.com, or bjhowa3@idaccr.org), and
Keith Frankston (kmwf1416@fastmail.fm, or k.frankston@idaccr.org)


This repository contains code to compare MCTS-UCB (AlphaZero style) and RMCTS (posterior / optimized policy) implementations and timings, plus tools to run inference with ONNX / TensorRT and PyCUDA. 

The new algorithm RMCTS is discussed in the paper, ./paper/rmcts.pdf

The RMCTS code (the focus of this git repo) can be found in src/c/rmcts.c

Quick highlights
- Research code comparing two MCTS variants (timings, experiments).
- Tools for running TensorRT / ONNX inference and measuring latency.

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


AFTER INSTALLING THE ABOVE PACKAGES:

To make a particular game, such as Othello:
$ make GAME=othello
this compiles relevant code and puts the results in ./build/othello

To make all three games, run 
$ ./build_all_games.sh 
It simply runs "make GAME=connect4; make GAME=dotbox; make GAME=othello;".

The perform your own timings, you can:
$ python othello_timings.py
$ python connect4_timings.py
$ python dotbox_timings.py

-----------------------------------------------------------------------------

Copyright (c) 2025, Institute for Defense Analyses, 730 Glebe Rd, Alexandria, VA 22305-3086; 703-845-2500

This material may be reproduced by or for the U.S. Government pursuant to all applicable FAR and DFARS clauses.