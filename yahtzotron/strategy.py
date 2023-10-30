import itertools
import os
import pickle
from collections import Counter
from functools import lru_cache
from math import factorial

import numpy as np
import tqdm


@lru_cache(maxsize=None)
def _memoized_score(roll, cat_idx, scorecard, scores, ruleset):
    dice_count = np.bincount(roll, minlength=7)[1:]
    return ruleset.score(dice_count, cat_idx, scorecard, scores)


@lru_cache(maxsize=None)
def num_unique_permutations(tup):
    item_count = Counter(tup)
    norm_factor = 1
    for count in item_count.values():
        norm_factor *= factorial(count)
    return factorial(len(tup)) // norm_factor


def get_expected_reward(initial_roll, cat_idx, ruleset):
    num_dice = ruleset.num_dice
    dummy_scorecard = (0,) * ruleset.num_categories
    dummy_scores = (0,) * ruleset.num_categories

    expected_reward = {}
    for action in itertools.product((0, 1), repeat=num_dice):
        kept_dice = tuple(die for die, kept in zip(initial_roll, action) if kept)
        num_rerolled = num_dice - len(kept_dice)

        count = 0
        expected_reward[action] = 0.0
        for final_roll in itertools.combinations_with_replacement(
            range(1, 7), num_rerolled
        ):
            possible_combinations = num_unique_permutations(final_roll)
            final_dice = tuple(sorted(kept_dice + final_roll))
            expected_reward[action] += possible_combinations * _memoized_score(
                final_dice, cat_idx, dummy_scorecard, dummy_scores, ruleset
            )
            count += possible_combinations

        expected_reward[action] /= count

    return expected_reward


def assemble_roll_lut(ruleset):
    """Assemble look up table of expected rewards for given ruleset.

    This enumerates all possible outcomes of each decision.

    Returns a dict

    {
        "full": Nested dict with expected reward for each roll, category, keep action.
        "marginal-0": Dict with expected reward for each category, marginalized over all
            rolls if all dice are kept.
        "marginal-1": Dict with expected reward for each category, marginalized over all
            rolls if the best keep action is chosen.
    }
    """
    num_dice = ruleset.num_dice

    # loop over all possible unique rolls with n 6-sided dice
    roll_combinations = itertools.combinations_with_replacement(range(1, 7), num_dice)
    total_elements = int(factorial(5 + num_dice) / factorial(num_dice) / factorial(5))

    expected_reward = {}

    pbar = tqdm.tqdm(
        roll_combinations,
        desc="Pre-computing payoff LUT... " + "🎲" * num_dice,
        total=total_elements,
    )

    for initial_roll in pbar:
        expected_reward[initial_roll] = {}
        for cat_idx in range(ruleset.num_categories):
            expected_reward[initial_roll][cat_idx] = get_expected_reward(
                initial_roll, cat_idx, ruleset
            )

    # assemble marginal reward LUTs
    expected_marginal_reward_0step = {}
    expected_marginal_reward_1step = {}

    for cat_idx in range(ruleset.num_categories):
        expected_marginal_reward_0step[cat_idx] = 0
        expected_marginal_reward_1step[cat_idx] = 0

        count = 0
        for roll, reward in expected_reward.items():
            multiplier = num_unique_permutations(roll)
            expected_marginal_reward_0step[cat_idx] += (
                multiplier * reward[cat_idx][(1,) * num_dice]
            )
            expected_marginal_reward_1step[cat_idx] += multiplier * max(
                reward[cat_idx].values()
            )
            count += multiplier

        expected_marginal_reward_0step[cat_idx] /= count
        expected_marginal_reward_1step[cat_idx] /= count

    return {
        "full": expected_reward,
        "marginal-0": expected_marginal_reward_0step,
        "marginal-1": expected_marginal_reward_1step,
    }

def create_lut(path, ruleset):
    """Load cached look-up table, or compute from scratch."""
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        roll_lut = assemble_roll_lut(ruleset)
        with open(path, "wb") as f:
            pickle.dump(roll_lut, f)

    with open(path, "rb") as f:
        roll_lut = pickle.load(f)

    return roll_lut
