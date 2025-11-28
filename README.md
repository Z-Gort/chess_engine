# Chess Engine

An AlphaZero-style chess engine combining a DNN with my own implementation of Monte Carlo Tree Search (MCTS).

## Overview

This engine uses a convolutional neural network with residual blocks to evaluate positions and predict moves, guided by MCTS for search. The architecture features separate policy and value heads, similar to AlphaGo Zero.

## Architecture

- **Neural Network**: Residual CNN with configurable depth (10 or 20 blocks) and width (128 or 256 filters)
- **Search**: PUCT-based Monte Carlo Tree Search
- **Board Encoding**: 16-plane representation of chess positions
- **Policy Output**: 4608-dimensional vector covering all possible moves

## Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Dependencies are pre-installed in venv:
# - python-chess
# - torch
# - numpy
```

## Usage

**Play against the engine:**

```bash
python play_chess.py
```

Enter moves in UCI format (e.g., `e2e4`, `g1f3`).

## Files

- `AlphaZeroNetwork.py` - Neural network architecture
- `mcts.py` - MCTS implementation
- `encoder.py` - Board encoding/decoding
- `play_chess.py` - Interactive play interface
- `eval.py` - Position evaluation
- `uci.py` - UCI protocol (basic implementation)
- `weights/` - Pre-trained model weights

## Model Weights

Credits to https://github.com/jackdawkins11/pytorch-alpha-zero for the pretrained models/CNN architecture.

## Technical Details

The engine performs 1000 MCTS simulations per move by default, using the neural network to guide search with policy priors and value estimates. Search uses the PUCT formula with exploration constant C=1.5.
