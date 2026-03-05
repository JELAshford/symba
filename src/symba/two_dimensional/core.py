"""2d Barricelli"""

from matplotlib.animation import FuncAnimation
import matplotlib.pylab as plt
from itertools import product
from pathlib import Path
import numpy as np
import git

SIZE = 128
TIMESTEPS = 1024
MAX_VAL = 5
SEED = 1701


def tuple_add(t1, t2):
    return (t1[0] + t2[0], t1[1] + t2[1])


def tuple_sub(t1, t2):
    return (t1[0] - t2[0], t1[1] - t2[1])


def tuple_mul(t1, v: int):
    return (t1[0] * v, t1[1] * v)


def tuple_mod(t1, t2):
    return (t1[0] % t2[0], t1[1] % t2[1])


def gather_replication_candidate(state: list[list[tuple]]) -> list[set]:
    (h, w) = len(state), len(state[0])
    candidates = [[set() for _ in range(w)] for _ in range(h)]
    for pos in product(range(h), range(w)):
        current_val = state[pos[0]][pos[1]]
        values_checked = {(0, 0)}
        value_to_check = current_val
        while value_to_check not in values_checked:
            values_checked.add(value_to_check)
            target_position = tuple_mod(tuple_add(pos, value_to_check), (h, w))
            candidates[target_position[0]][target_position[1]].add(current_val)
            value_to_check = state[target_position[0]][target_position[1]]
    return candidates


def norm_zero(
    current_state: np.ndarray,
    collisions: list[set],
):
    h, w = len(current_state), len(current_state[0])
    coords = product(range(h), range(w))
    new_state = [[(0, 0)] * w for _ in range(h)]
    for y, x in coords:
        these_collisions = collisions[y][x]
        if len(these_collisions) == 1:
            new_state[y][x] = list(these_collisions)[0]
    return new_state


if __name__ == "__main__":
    # Get save path relative to project root
    project_root = Path(git.Repo(".", search_parent_directories=True).working_dir)
    SAVE_DIR = project_root / "out/two_dimensional"
    SAVE_DIR.mkdir(exist_ok=True, parents=True)

    # Basic baricelli copying
    rng = np.random.default_rng(seed=SEED)
    grid = np.zeros((TIMESTEPS, SIZE, SIZE, 2)).astype(int)
    grid[0, ...] = rng.integers(-MAX_VAL, MAX_VAL + 1, size=(SIZE, SIZE, 2))
    grid[0, ...] *= (rng.random((SIZE, SIZE, 2)) < 0.8).astype(int)
    grid = [
        [[tuple(map(int, s)) for s in row] for row in timestep] for timestep in grid
    ]

    # Iteratively apply the replication updates/mutation norms
    for step in range(1, TIMESTEPS):
        candidates = gather_replication_candidate(grid[step - 1])
        grid[step] = norm_zero(grid[step - 1], candidates)

    # View the full run
    # convert the 2d vectors into RGB image, where red is angle and green is magnitude
    angle_data = np.array(grid)
    frames = np.zeros(shape=(TIMESTEPS, SIZE, SIZE, 3))
    frames[..., 1] = (np.atan2(angle_data[..., 0], angle_data[..., 1]) + np.pi) / (
        2 * np.pi
    )
    frames[..., 2] = np.sqrt(angle_data[..., 0] ** 2 + angle_data[..., 1] ** 2) / (
        MAX_VAL * np.sqrt(2)
    )

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    ax.set_axis_off()
    im = ax.imshow(frames[1, ...], aspect="auto", interpolation="none")

    def animate(index):
        global frames
        im.set_data(frames[(index + 1) * 2, ...])
        return (im,)

    anim = FuncAnimation(
        fig, animate, frames=(len(frames) // 2) - 1, interval=50, blit=True
    )
    anim.save(f"{SAVE_DIR}/two_dimensional.mp4")

    # View timesteps
    num_samples = 5
    fig, ax = plt.subplots(1, num_samples, figsize=(10, 2))
    for ind, sample in enumerate(
        np.linspace(1, TIMESTEPS - 1, num_samples).astype(int)
    ):
        ax[ind].imshow(
            frames[sample, ...],
            aspect="auto",
            interpolation="none",
        )
        ax[ind].set_title(f"Time: {sample}")
        ax[ind].set_axis_off()
    plt.savefig(
        f"{SAVE_DIR}/two_dimensional_samples.png",
        bbox_inches="tight",
        transparent=True,
        dpi=200,
    )
    plt.close()
