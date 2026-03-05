"""Make a version of the simulation that can run live in a web browser"""

from core import gather_replication_candidate, norm_zero
from drawerer import Simulation, DisplayApp
import matplotlib.pylab as plt
import numpy as np


class OneDimBarricelli(Simulation):
    def __init__(self, size=256, time=128, max_val=5, seed=1701, **kwargs):
        super().__init__(**kwargs)
        self.timestep = 0
        self.size = size
        self.fifth = self.size // 5
        self.time = time
        self.max_val = max_val
        self.rng = np.random.default_rng(seed=seed)
        self.random_init()
        self.cmap = plt.get_cmap("bwr")
        self.num_substeps = 1

    def random_init(self):
        self.grid = np.zeros((self.time, self.size)).astype(int)
        self.grid[0, :] = self.rng.integers(
            -self.max_val, self.max_val + 1, size=(self.size)
        )
        self.grid[
            0,
            self.rng.choice(
                np.arange(self.size), size=int(self.size * (4 / 5)), replace=False
            ),
        ] = 0

    def step(self):
        while True:
            for _subset in range(self.num_substeps):
                next_timestep = self.timestep + 1
                if next_timestep >= self.time:
                    self.timestep = 0
                    next_timestep = 1
                    self.random_init()
                candidates = gather_replication_candidate(self.grid[self.timestep, :])
                new_state = norm_zero(self.grid[self.timestep, :], candidates)
                self.grid[next_timestep, :] = new_state
                self.timestep = next_timestep

            normed_grid = ((self.grid + self.max_val) / (2 * self.max_val)) * 255
            # draw_grid = repeat(normed_grid, "h w -> h w 3").astype(np.uint8)
            draw_grid = (self.cmap(normed_grid.astype(np.uint8)) * 255)[..., :3].astype(
                np.uint8
            )
            yield draw_grid


if __name__ == "__main__":
    DisplayApp(OneDimBarricelli(), site_path="./test", target_fps=60).run()
