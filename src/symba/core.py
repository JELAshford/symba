import numpy as np


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
    # Create a set of candidates for each position
    candidates = [set() for _ in range(num_states)]
    for pos, current_val in enumerate(state):
        # Store all offsets already checked
        offsets_checked = {current_val}
        offset_to_check = current_val
        while offset_to_check != 0:
            # Get target position and it's value
            target_position = (pos + offset_to_check) % num_states
            target_value = state[target_position]
            # Add original value as a candidate for this position
            candidates[target_position].add(current_val)
            # Decide whether to continue searching
            if target_value not in offsets_checked:
                offsets_checked.add(target_value)
                offset_to_check = target_value
            else:
                offset_to_check = 0
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
    return np.array([list(vals)[0] if len(vals) == 1 else 0 for vals in candidates])


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
