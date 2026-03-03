import matplotlib.pylab as plt
from pathlib import Path
import numpy as np
import git


def gather_replication_candidate(state: np.ndarray) -> list[set]:
    """Gather the replication candidate values at each position in state after all positions attempt replication.

    For each currently non-zero valued position in the state, it's value becomes a "replication candidate" for a
    target position, equal to the current position plus the current value.

    If that target position is occupied by a non-zero value in the current state, the original value will also
    becomes a "replication candidate" for a new target position equal to it's current position plus the value in the
    current target position.

    This process is repeated until the new target position is occupied by an already encountered position or a zero.

    e.g. a 4 at position 2 will become a replication candidate for position 6 (2 + 4). If position 6 is occupied by a
    -1 in the current state, the 4 will also become a replication candidate for position 1 (2 + (-1)). If position 1
    is occupied by a 0, 4, or -1 the replication attempts terminate.

    Parameters
    ---------
    state
        Current 1D state of all positions, shape=(num_states,)

    Returns
    -------
    candidates
        List of replication candidate values for each position in the 1D state

    """
    (num_states,) = state.shape
    candidates = [set() for _ in range(num_states)]
    for pos, current_val in enumerate(state):
        values_checked = {0}
        value_to_check = current_val
        while value_to_check not in values_checked:
            values_checked.add(value_to_check)
            target_position = (pos + value_to_check) % num_states
            candidates[target_position].add(current_val)
            value_to_check = state[target_position]
    return candidates


def norm_zero(
    state: np.ndarray,
    candidates: list[set],
) -> int:
    """Barricelli's 0 Norm.
    Resolve replication candidates by setting state to 0 in all cases except only 1 candidate.

    Parameters
    ----------
    state:
        Current 1D state of all positions, shape=(num_states,) [Not used in this `norm`]
    candidates:
        Replication candidates for each position in state (see `gather_replication_candidate` for details.)

    Returns
    -------
    new_state:
        New 1D state after resolving replication behaviour
    """
    return np.array([vals.pop() if len(vals) == 1 else 0 for vals in candidates])


def norm_A(
    current_state: np.ndarray,
    candidates: list[set],
):
    # If no collision, return the value that got written to that position
    size = len(current_state)
    new_state = np.zeros_like(current_state)
    for ind, (candidates, current_val) in enumerate(zip(candidates, current_state)):
        if len(candidates) == 1:
            new_state[ind] = list(candidates)[0]
        elif len(candidates) == 0:
            continue
        elif current_val != 0:
            continue
        else:
            # There is a collision
            # search left for nearest active value
            left_dist = 1
            left_pos = (ind - 1) % size
            while (left_val := current_state[left_pos]) == 0:
                left_pos = (left_pos - 1) % size
                left_dist += 1

            right_dist = 1
            right_pos = (ind + 1) % size
            while (right_val := current_state[right_pos]) == 0:
                right_pos = (right_pos + 1) % size
                right_dist += 1

            if np.sign(left_val) == np.sign(right_val):
                new_state[ind] = left_dist + right_dist
            else:
                new_state[ind] = -(left_dist + right_dist)
    return new_state


def norm_B(
    current_state: np.ndarray,
    candidates: list[set],
):
    # If no collision, return the value that got written to that position
    size = len(current_state)
    new_state = np.zeros_like(current_state)
    for ind, (candidates, current_val) in enumerate(zip(candidates, current_state)):
        if len(candidates) == 1:
            new_state[ind] = list(candidates)[0]
        elif len(candidates) == 0:
            continue
        elif current_val != 0:
            continue
        else:
            # There is a collision
            # search left for nearest active value
            left_dist = 1
            left_pos = (ind - 1) % size
            while (left_val := current_state[left_pos]) == 0:
                left_pos = (left_pos - 1) % size
                left_dist += 1

            right_dist = 1
            right_pos = (ind + 1) % size
            while (right_val := current_state[right_pos]) == 0:
                right_pos = (right_pos + 1) % size
                right_dist += 1

            if np.sign(left_val) == np.sign(right_val):
                new_state[ind] = left_dist + right_dist - 1
            else:
                new_state[ind] = -(left_dist + right_dist - 1)
    return new_state


def norm_C(
    current_state: np.ndarray,
    candidates: list[set],
):
    # If no collision, return the value that got written to that position
    size = len(current_state)
    new_state = np.zeros_like(current_state)
    for ind, (candidates, current_val) in enumerate(zip(candidates, current_state)):
        if len(candidates) == 1:
            new_state[ind] = list(candidates)[0]
        elif len(candidates) == 0:
            continue
        else:
            # There is a collision
            # search left for nearest active value
            left_pos = (ind - 1) % size
            while (left_val := current_state[left_pos]) == 0:
                left_pos = (left_pos - 1) % size

            right_pos = (ind + 1) % size
            while (right_val := current_state[right_pos]) == 0:
                right_pos = (right_pos + 1) % size

            new_state[ind] = left_val - right_val
    return new_state


def norm_D(
    current_state: np.ndarray,
    candidates: list[set],
):
    # If no collision, return the value that got written to that position
    size = len(current_state)
    new_state = np.zeros_like(current_state)
    for ind, (candidates, current_val) in enumerate(zip(candidates, current_state)):
        if len(candidates) == 1:
            new_state[ind] = list(candidates)[0]
        elif len(candidates) == 0:
            continue
        elif current_val != 0:
            continue
        else:
            # There is a collision
            left_val = current_state[(ind - current_val) % size]
            right_val = current_state[(ind + current_val) % size]
            if left_val == right_val:
                new_state[ind] = -current_val + 2 * right_val
            else:
                continue
    return new_state


if __name__ == "__main__":
    SIZE = 512
    TIMESTEPS = 128
    MAX_VAL = 2
    SEED = 1701

    # Get save path relative to project root
    project_root = Path(git.Repo(".", search_parent_directories=True).working_dir)
    SAVE_DIR = project_root / "out/one_dimensional"
    SAVE_DIR.mkdir(exist_ok=True, parents=True)

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
    plt.savefig(
        f"{SAVE_DIR}/one_dimensional.png",
        bbox_inches="tight",
        transparent=True,
        dpi=300,
    )
    plt.close()
