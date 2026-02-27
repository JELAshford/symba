from symba.core import gather_replication_candidate, norm_zero
import matplotlib.pylab as plt
import numpy as np

SIZE = 512
TIMESTEPS = 128
MAX_VAL = 2
SEED = 1701

# Basic baricelli copying
rng = np.random.default_rng(seed=SEED)
grid = np.zeros((TIMESTEPS, SIZE)).astype(int)

# Initialise with sparse random
grid[0, :] = rng.integers(-MAX_VAL, MAX_VAL + 1, size=(SIZE))
grid[0, rng.choice(np.arange(SIZE), size=int(SIZE * (4 / 5)), replace=False)] = 0

# Iteratively apply the replication updates/mutation norms
for step in range(1, TIMESTEPS):
    candidates = gather_replication_candidate(grid[step - 1, :])
    grid[step, :] = norm_zero(grid[step - 1, :], candidates)

# View the final state!
fig, ax = plt.subplots(1, 1)
ax.imshow(grid, aspect="auto", interpolation="none", cmap="bwr")
plt.axis("off")
plt.savefig("out/one_dimensional.png", bbox_inches="tight", transparent=True, dpi=300)
plt.close()
