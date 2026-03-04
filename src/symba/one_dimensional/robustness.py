"""Test the robustness of the organism by colliding it with a number. Figure 15"""

from symba.one_dimensional.core import gather_replication_candidate, norm_zero
import matplotlib.pylab as plt
from pathlib import Path
import numpy as np
import git


# Generate examples with different random seeds
SIZE = 500
TIMESTEPS = 128
MAX_VAL = 5
SEED = 1701

# Get save path relative to project root
project_root = Path(git.Repo(".", search_parent_directories=True).working_dir)
SAVE_DIR = project_root / "out/one_dimensional"
SAVE_DIR.mkdir(exist_ok=True, parents=True)

# Generate random attackers
rng = np.random.default_rng(seed=SEED)
attackers = ((100, 2), (100, 5), (100, 10), (400, -1), (400, -3), (400, -6))

fig, axs = plt.subplots(2, len(attackers) // 2)
for (position, value), ax in zip(attackers, axs.flatten()):
    grid = np.zeros((TIMESTEPS, SIZE)).astype(int)

    # Initialise with organism and attacker
    grid[0, 240:260] = [0, 0, 5, 0, 0, 0, 5, 0, 1, -3, 1, -3, 0, 0, 0, 0, 0, 0, 0, 0]
    grid[0, position] = value

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
    f"{SAVE_DIR}/robustness.png",
    bbox_inches="tight",
    transparent=True,
    dpi=300,
)
plt.close()
