# Symba (**Sym**biogenetic **ba**rricelli)

Experiments with some of the systems described by Niels Barricelli, with a focus on the emergeny symbiogenetic behaviour.

This repo includes implementations of the 1- and 2-D variants of Barricelli's automata, and an implentation of a DNA-like system that follows similar principles.

## Barricellii Automata Overview

The Barricelli systems are broken into a two-phase loop, where given the **state** of the system the program:

1. Gathers all replication candidates for each position in the state, driven by the offsets/jumps defined by the values in every other state.

2. Resolves replication conflicts (places where there are > 1 replication candidates) by applying a mutation rule or "norm".

## Repo Overview

The source code for these experiments are stored in `src/symba/` which is divided by experiment type:

- `one_dimensional` holds the implementation and experiments with 1-D Barricelli automata.-
- `two_dimensional` holds our preliminary implementation of 2-D Barricelli automata which is functional but unfortunately rather slow!
- `dna` has our experiments with Barricellis `DNA norms`.

## Running the Experiments

This project is managed with `uv`, and the environment can be setup with `uv sync`. To run the individual scripts you can use `uv run src/symba/path/to/file.py`, and outputs will be generated in the `out` folder.
