# Symba

Implement the automata described by Niels Barricelli, with a focus on the emergeny symbiogenetic behaviour (**Sym**biogenetic **ba**rricelli).

## Overview

The Barricelli systems are broken into a two-phase loop, where given the **state** of the system the program:

1. Gathers all replication candidates for each position in the state, driven by the offsets/jumps defined by the values in every other state.
2. Resolves replication conflicts (places where there are > 1 replication candidates) by applying a mutation rule or "norm".

## Running

This project is managed with `uv`, and the environment can be setup with `uv sync`.

To the run the demo 1D automata, you can run: `uv run src/symba/demo.py`
