"""Examples of running norm zero, currently Figure 10 in the paper"""

from symba.one_dimensional.core import gather_replication_candidate, norm_zero
import matplotlib.pylab as plt
from pathlib import Path
import numpy as np
import git


# Generate examples with different random seeds
SIZE = 256
TIMESTEPS = 64
MAX_VAL = 2
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
    ax.imshow(grid, aspect="auto", interpolation="none", cmap="bwr")
    ax.set_axis_off()

plt.savefig(
    f"{SAVE_DIR}/norm_zero_example.png",
    bbox_inches="tight",
    transparent=True,
    dpi=300,
)
plt.close()
