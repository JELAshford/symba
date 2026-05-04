"""Implement some demos inspired by the papers and previous work of Barricelli."""

from symba.one_dimensional.core import (
    gather_replication_candidate,
    norm_zero,
    norm_A,
    norm_B,
    norm_C,
    norm_D,
)
import matplotlib.pylab as plt
from pathlib import Path
import numpy as np
import git


# Helper functions for setup/plotting
def save_state(grid: np.ndarray, save_path: Path, cmap: str = "gray") -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(grid, aspect="auto", interpolation="none", cmap=cmap)
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", transparent=True, dpi=200)
    plt.close()


# Get save path relative to project root
project_root = Path(git.Repo(".", search_parent_directories=True).working_dir)
SAVE_DIR = project_root / "out/one_dimensional/demos"


# Show off each rule
TIMESTEPS = 512
MAX_VAL = 10
SIZE = 512
SEED = 8346

rng = np.random.default_rng(seed=SEED)
start_grid = np.zeros((TIMESTEPS, SIZE)).astype(int)
start_grid[0, :] = rng.integers(-MAX_VAL, MAX_VAL + 1, size=(SIZE))
start_grid[0, rng.choice(np.arange(SIZE), size=int(SIZE * (4 / 5)), replace=False)] = 0

rules = {
    "normA": norm_A,
    "normB": norm_B,
    "normC": norm_C,
    "normD": norm_D,
    "norm_zero": norm_zero,
}
for name, rule in rules.items():
    this_grid = start_grid.copy()
    for step in range(1, TIMESTEPS):
        candidates = gather_replication_candidate(this_grid[step - 1, :])
        this_grid[step, :] = rule(this_grid[step - 1, :], candidates)
    save_state(this_grid, Path(f"{SAVE_DIR}/{name}.png"), cmap="bwr")


# Figure 17
timesteps = 30
grid = np.zeros((timesteps, 60)).astype(int)
grid[0, :] = (
    [0] * 20
    + [0, 0, 0, 0, 0, 1, -2, 1, 1, -2, 0, 1, -2, 1, 1, -2, 0, 0, 0, 0]
    + [0] * 20
)
for step in range(1, timesteps):
    candidates = gather_replication_candidate(grid[step - 1, :])
    grid[step, :] = norm_zero(grid[step - 1, :], candidates)
    grid[:, :20] = 0
    grid[:, -20:] = 0
save_state(grid, Path(f"{SAVE_DIR}/fig17.png"))


# Figure 5
timesteps = 20
grid = np.zeros((timesteps, 60)).astype(int)
grid[0, :] = (
    [0] * 20 + [0, 0, 5, 0, 0, 0, 5, 0, 1, -3, 1, -3, 0, 0, 0, 0, 0, 0, 0, 0] + [0] * 20
)
for step in range(1, timesteps):
    candidates = gather_replication_candidate(grid[step - 1, :])
    grid[step, :] = norm_zero(grid[step - 1, :], candidates)
    grid[:, :20] = 0
    grid[:, -20:] = 0
save_state(grid, Path(f"{SAVE_DIR}/fig5.png"))


# Figure 3: emergent crossover
timesteps = 20
edge = 200
grid = np.zeros((timesteps, 100 + (edge * 2))).astype(int)
left_parent = [5, -11, 1, -3] * 6
right_parent = [9, -11, 1, -7] * 7
grid[0, :] = (
    [0] * (10 + edge) + left_parent + [0] * 16 + right_parent + [0] * (22 + edge)
)
for step in range(1, timesteps):
    candidates = gather_replication_candidate(grid[step - 1, :])
    grid[step, :] = norm_zero(grid[step - 1, :], candidates)
save_state(grid, Path(f"{SAVE_DIR}/fig3.png"))
breakpoint()

with open("file", "w") as f:
    for row in grid:
        line = " ".join(map(lambda s: str(s).zfill(2), row))
        f.write(line + "\n")
