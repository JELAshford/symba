"""Make a version of the simulation that can run live in a web browser"""

from symba.two_dimensional.core import gather_replication_candidate, norm_zero
from drawerer import Simulation, DisplayApp
import matplotlib.pylab as plt
from einops import repeat
import numpy as np


class TwoDimBarricelli(Simulation):
    def __init__(self, size=128, max_val=5, seed=1701, **kwargs):
        super().__init__(**kwargs)
        self.timestep = 0
        self.size = size
        self.max_val = max_val
        self.rng = np.random.default_rng(seed=seed)
        self.grid = self.random_init()

    def random_init(self):
        grid = self.rng.integers(
            -self.max_val, self.max_val + 1, size=(self.size, self.size, 2)
        )
        grid *= (self.rng.random((self.size, self.size, 2)) < 0.8).astype(int)
        grid = [[tuple(map(int, s)) for s in row] for row in grid]
        return grid

    def step(self):
        while True:
            candidates = gather_replication_candidate(self.grid)
            self.grid = norm_zero(self.grid, candidates)

            angle_data = np.array(self.grid)
            draw_grid = np.zeros((self.size, self.size, 3)) * 0.4
            draw_grid[..., 1] = (
                np.atan2(angle_data[..., 0], angle_data[..., 1]) + np.pi
            ) / (2 * np.pi)
            draw_grid[..., 2] = np.sqrt(
                angle_data[..., 0] ** 2 + angle_data[..., 1] ** 2
            ) / (self.max_val * np.sqrt(2))
            yield ((draw_grid**0.5) * 255).astype(np.uint8)


if __name__ == "__main__":
    DisplayApp(TwoDimBarricelli(), site_path="./test", target_fps=30).run()
