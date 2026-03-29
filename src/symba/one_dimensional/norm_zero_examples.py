"""Examples of running norm zero, currently Figure 10 in the paper"""

from pathlib import Path

import git
import matplotlib.pylab as plt
import numpy as np
from symba.one_dimensional.core import gather_replication_candidate, norm_zero

# Generate examples with different random seeds
SIZE = 512
TIMESTEPS = 128
MAX_VAL = 5
SEEDS = (1701, 1298, 124710, 10941, 127912, 987)

# Get save path relative to project root
project_root = Path(git.Repo(".", search_parent_directories=True).working_dir)
SAVE_DIR = project_root / "out/one_dimensional"
SAVE_DIR.mkdir(exist_ok=True, parents=True)

fig, axs = plt.subplots(2, len(SEEDS) // 2)
for seed, ax in zip(SEEDS, axs.flatten()):
    rng = np.random.default_rng(seed=seed)
    grid = np.zeros((TIMESTEPS, SIZE)).astype(int)

    # Initialise with sparse random
    grid[0, :] = rng.integers(-MAX_VAL, MAX_VAL + 1, size=(SIZE))
    grid[0, rng.choice(np.arange(SIZE), size=int(SIZE * (4 / 5)), replace=False)] = 0

    # Iteratively apply the replication updates/mutation norms
    for step in range(1, TIMESTEPS):
        candidates = gather_replication_candidate(grid[step - 1, :])
        grid[step, :] = norm_zero(grid[step - 1, :], candidates)

    # Draw to grid
    ax.imshow(
        grid,
        aspect="auto",
        interpolation="none",
        cmap="bwr",
        vmin=-MAX_VAL,
        vmax=MAX_VAL,
    )
    ax.set_axis_off()

plt.tight_layout()
plt.savefig(
    f"{SAVE_DIR}/norm_zero_example.png",
    bbox_inches="tight",
    transparent=True,
    dpi=300,
)
plt.close()
