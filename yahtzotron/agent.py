import functools
import os
import pickle

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey
from loguru import logger

from .rulesets import AVAILABLE_RULESETS
from .strategy import create_lut

TRANSLATE_FROM_ORIGINAL_DECIDE = {'linear': 'Dense_0',
                                  'linear_1': 'Dense_1',
                                  'linear_2': 'Dense_2',
                                  'linear_3': 'Value',
                                  'linear_4': 'Keep',
                                  'linear_5': 'Category'}
TRANSLATE_FROM_ORIGINAL_STRATEGY = {'linear_3': 'Categories'}

module_key = __name__ + '$params'  # TODO: should module name be accepted?

memoize = functools.lru_cache(maxsize=None)

DISK_CACHE = os.path.expanduser(os.path.join("~", ".yahtzotron"))


@memoize
def create_network(objective, num_dice, num_categories):
    """Create the neural networks used by the agent."""
    from yahtzotron.training import MINIMUM_LOGIT

    input_groups = [
        1,  # number of rerolls left
        6,  # count of each die value
        num_categories,  # player scorecard
        2,  # player upper and lower scores
    ]

    if objective == "win":
        input_groups.append(
            1,  # opponent value
        )

    input_shape = (1, sum(input_groups))

    keep_action_space = 2 ** num_dice

    class Decide(nn.Module):
        @nn.compact
        def __call__(self, inputs):
            rolls_left = inputs[..., 0, None]
            player_scorecard_idx = slice(sum(input_groups[:2]), sum(input_groups[:3]))
            init = jax.nn.initializers.variance_scaling(2.0, "fan_in", "truncated_normal")

            x = nn.Dense(128, kernel_init=init)(inputs)
            x = jax.nn.relu(x)
            x = nn.Dense(256, kernel_init=init)(x)
            x = jax.nn.relu(x)
            x = nn.Dense(128, kernel_init=init)(x)
            x = jax.nn.relu(x)

            out_value = nn.Dense(1, name="Value")(x)

            out_keep = nn.Dense(keep_action_space, name="Keep")(x)

            out_category = nn.Dense(num_categories, name="Category")(x)
            out_category = jnp.where(
                # disallow already filled categories
                inputs[..., player_scorecard_idx] == 1,
                -jnp.inf,
                out_category,
            )

            if keep_action_space < num_categories:
                out_keep = Decide.pad_(out_keep, num_categories - keep_action_space)

            elif keep_action_space > num_categories:
                out_category = Decide.pad_(out_category, keep_action_space - num_categories)

            out_action = jnp.where(rolls_left == 0, out_category, out_keep)

            return out_action, jnp.squeeze(out_value, axis=-1)

        @staticmethod
        def pad_(logit, num_pad):
            pad_shape = [(0, 0)] * (logit.ndim - 1) + [(0, num_pad)]
            return jnp.pad(logit, pad_shape, constant_values=MINIMUM_LOGIT)

    class Strategy(nn.Module):
        @nn.compact
        def __call__(self, inputs):
            player_scorecard_idx = slice(sum(input_groups[:2]), sum(input_groups[:3]))
            # init = nn.initializers.variance_scaling(2.0, "fan_in", "truncated_normal")

            x = nn.Dense(64)(inputs)
            x = jax.nn.relu(x)
            x = nn.Dense(128)(x)
            x = jax.nn.relu(x)
            x = nn.Dense(64)(x)
            x = jax.nn.relu(x)

            out_category = nn.Dense(num_categories, name="Categories")(x)
            out_category = jnp.where(
                # disallow already filled categories
                inputs[..., player_scorecard_idx] == 1,
                MINIMUM_LOGIT,
                out_category,
            )

            return out_category

    forward = Decide()
    forward_strategy = Strategy()

    return {
        "input-shape": input_shape,
        "network": forward,
        "strategy-network": forward_strategy,
    }


def get_opponent_value(opponent_scorecards, network, weights):
    """Compute the maximum value of all given opponents."""
    if not hasattr(opponent_scorecards, "__iter__"):
        opponent_scorecards = [opponent_scorecards]

    net_input = np.stack(
        [
            assemble_network_inputs(2, np.zeros(6), o.to_array(), 0.0)
            for o in opponent_scorecards
        ],
        axis=0,
    )
    _, opponent_values = network(weights, inputs=net_input)
    return np.max(opponent_values)


def play_turn(
    player_scorecard,
    objective,
    network,
    weights,
    num_dice,
    use_lut=None,
    opponent_value=None,
):
    """Play a turn.

    This is a generator that yields a dict describing the current state
    after each action.
    """
    player_scorecard_arr = player_scorecard.to_array()

    if opponent_value is None and objective == "win":
        raise ValueError("opponent value must be given for win objective")

    current_dice = (0,) * num_dice
    dice_to_keep = (0,) * num_dice

    for rolls_left in (2, 1, 0):
        kept_dice = tuple(die for die, keep in zip(current_dice, dice_to_keep) if keep)
        roll_input = yield kept_dice
        current_dice = tuple(sorted(roll_input))
        dice_count = np.bincount(current_dice, minlength=7)[1:]
        assert bool(np.all(dice_count - np.bincount(kept_dice, minlength=7)[1:] >= 0))

        if use_lut:
            observation = assemble_network_inputs(
                rolls_left, dice_count, player_scorecard_arr, opponent_value
            )
            action = get_action_greedy(
                rolls_left, current_dice, player_scorecard, use_lut
            )
            value = None
        else:
            observation, action, value = get_action(
                rolls_left,
                dice_count,
                player_scorecard_arr,
                opponent_value,
                network,
                weights,
            )

        if rolls_left > 0:
            keep_action = action
            category_idx = None
            dice_to_keep = np.unpackbits(
                np.uint8(keep_action), count=num_dice, bitorder="little"
            )
        else:
            keep_action = None
            category_idx = action
            dice_to_keep = [1] * num_dice

        logger.debug(" Observation: {}", observation)
        logger.debug(" Keep action: {}", dice_to_keep)
        logger.debug(" Value: {}", value)

        yield dict(
            rolls_left=rolls_left,
            observation=observation,
            action=action,
            keep_action=keep_action,
            category_idx=category_idx,
            value=value,
            dice_count=dice_count,
            dice_to_keep=dice_to_keep,
        )

    cat_name = player_scorecard.ruleset_.categories[category_idx].name
    logger.info(
        "Final roll: {} | Picked category: {} ({})",
        current_dice,
        category_idx,
        cat_name,
    )


def assemble_network_inputs(
    rolls_left, dice_count, player_scorecard, opponent_value=None
):
    """Re-shape inputs to be used as network input."""
    inputs = [
        np.asarray([rolls_left]),
        dice_count,
        player_scorecard,
    ]
    if opponent_value is not None:
        inputs.append(np.asarray([opponent_value]))

    return np.concatenate(inputs)


def get_action(
    rolls_left,
    dice_count,
    player_scorecard,
    opponent_value,
    network,
    weights,
):
    """Get an action according to current policy."""

    def choose_from_logits(logits):
        # pure NumPy version of jax.random.categorical
        logits = np.asarray(logits)
        prob = np.exp(logits - logits.max())
        prob /= prob.sum()
        return np.random.choice(logits.shape[0], p=prob)

    network_inputs = assemble_network_inputs(
        rolls_left, dice_count, player_scorecard, opponent_value
    )
    action_logits, value = network(weights, network_inputs)
    action = choose_from_logits(action_logits)

    return network_inputs, action, value


def get_action_greedy(rolls_left, current_dice, player_scorecard, roll_lut):
    """Get an action according to look-up table.

    This greedily picks the action with the highest expected reward advantage.
    This is far from optimal play but should be a good baseline.
    """
    num_dice = player_scorecard.ruleset_.num_dice
    num_categories = player_scorecard.ruleset_.num_categories

    if rolls_left > 0:
        best_payoff = lambda lut, cat: max(lut[cat].values())
        marginal_lut = roll_lut["marginal-1"]
    else:
        # there is no keep action, so only keeping all dice counts
        best_payoff = lambda lut, cat: lut[cat][(1,) * num_dice]
        marginal_lut = roll_lut["marginal-0"]

    expected_payoff = [
        (
            best_payoff(roll_lut["full"][current_dice], c)
            if player_scorecard.filled[c] == 0
            else -float("inf")
        )
        - marginal_lut[c]
        for c in range(num_categories)
    ]
    category_action = np.argmax(expected_payoff)

    if rolls_left > 0:
        category_ = roll_lut["full"][current_dice][category_action]
        dice_to_keep = max(
            category_.keys(),
            key=lambda k: category_[k],
        )
        keep_action = int(np.packbits(dice_to_keep, bitorder="little"))
        return keep_action

    return category_action


class Yahtzotron:
    def __init__(self, ruleset=None, load_path=None, rngs: PRNGKey = None, objective="win", greedy=False):
        if ruleset is None and load_path is None:
            raise ValueError("Either ruleset or load_path must be given")

        if rngs is None and not greedy:
            raise ValueError("rngs must be given, all Models require it")
        rngs, key1, key2 = jax.random.split(rngs, 3)

        possible_objectives = ("win", "avg_score")
        if objective not in possible_objectives:
            raise ValueError(
                f"Got unexpected objective {objective}, must be one of {possible_objectives}"
            )

        if load_path is None:
            self._ruleset = AVAILABLE_RULESETS[ruleset]
            self._objective = objective
        else:
            self.load(load_path)

        self._be_greedy = greedy
        self._roll_lut_path = os.path.join(
            DISK_CACHE, f"roll_lut_{self._ruleset.name}.pkl"
        )

        num_dice, num_categories = (
            self._ruleset.num_dice,
            self._ruleset.num_categories,
        )

        networks = create_network(self._objective, num_dice, num_categories)
        self._network = jax.jit(networks["network"].apply)
        self._strategy_network = jax.jit(networks["strategy-network"].apply)

        if load_path is None:
            self._weights = networks["network"].init(
                rngs=key1,
                inputs=jnp.empty(networks["input-shape"], dtype=jnp.float32),
            )
            self._strategy_weights = networks["strategy-network"].init(
                rngs=key2,
                inputs=jnp.empty(networks["input-shape"], dtype=jnp.float32),
            )

    def ruleset(self):
        return self._ruleset

    def compiled_network(self):
        return self._network

    def is_objective_winning(self):
        return self._objective == "win"

    def compiled_strategy_network(self, weights, observation):
        return self._strategy_network(weights, inputs=observation)

    def explain(self, observation):
        """Return the estimated probability for each final category action.

        This only makes sense if rolls_left > 0.
        """
        cat_logits = self._strategy_network(self._strategy_weights, inputs=observation)
        cat_prob = np.exp(cat_logits - cat_logits.max())
        cat_prob /= cat_prob.sum()
        return dict(sorted(enumerate(cat_prob), key=lambda k: k[1], reverse=True))

    def turn(
        self,
        player_scorecard,
        opponent_scorecards=None,
    ):
        """Play a turn.

        This is a generator that yields a dict describing the current state
        after each action.

        Opponent scorecards only have meaning if the objective is "win".
        """
        if self.is_objective_winning():
            if opponent_scorecards is None or (
                hasattr(opponent_scorecards, "__iter__")
                and len(opponent_scorecards) == 0
            ):
                opponent_scorecards = player_scorecard

            opponent_value = get_opponent_value(
                opponent_scorecards, self._network, self._weights
            )
        else:
            opponent_value = None

        if self._be_greedy:
            roll_lut = self._ruleset.get_lut()
        else:
            roll_lut = None

        yield from play_turn(
            player_scorecard,
            opponent_value=opponent_value,
            objective=self._objective,
            network=self._network,
            weights=self._weights,
            num_dice=self._ruleset.num_dice,
            use_lut=roll_lut,
        )

    def get_weights(self, strategy=False):
        """Get current network weights."""
        if strategy:
            return self._strategy_weights

        return self._weights

    def set_weights(self, new_weights, strategy=False):
        """Set current network weights."""
        new_weights = Yahtzotron.to_immutable_dict_(new_weights)
        if strategy:
            self._strategy_weights = new_weights
        else:
            self._weights = new_weights

    @staticmethod
    def to_immutable_dict_(new_weights):
        params_flat, tree_def = jax.tree_util.tree_flatten(new_weights)
        return jax.tree_util.tree_unflatten(tree_def, params_flat)

    def clone(self, rngs=None, keep_weights=True, greedy=False):
        """Create a copy of the current agent."""
        yzt = self.__class__(ruleset=self._ruleset.name, rngs=rngs, objective=self._objective, greedy=greedy)

        if keep_weights:
            yzt.set_weights(self.get_weights())

        return yzt

    def save(self, path):
        """Save agent to given pickle file."""
        os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)

        statedict = dict(
            objective=self._objective,
            ruleset=self._ruleset.name,
            weights=self._weights,
            strategy_weights=self._strategy_weights,
        )

        with open(path, "wb") as f:
            pickle.dump(statedict, f)

    def load(self, path):
        """Load agent from given pickle file."""
        with open(path, "rb") as f:
            statedict = pickle.load(f)

        self._objective = statedict["objective"]
        self._ruleset = AVAILABLE_RULESETS[statedict["ruleset"]]
        self._weights = Yahtzotron._translate_weights(statedict["weights"],
                                                      translation=TRANSLATE_FROM_ORIGINAL_DECIDE)
        self._strategy_weights = Yahtzotron._translate_weights(statedict["strategy_weights"],
                                                               translation=TRANSLATE_FROM_ORIGINAL_STRATEGY)

    @staticmethod
    def _translate_weights(weights, translation=None):
        if translation is None:
            translation = dict()
        if 'params' in weights:
            return weights
        return {'params': Yahtzotron._translate_parameters(translation,
                                                           weights)}

    @staticmethod
    def _translate_parameters(lookup, weights):
        if not isinstance(weights, dict):
            return weights
        params = dict()
        for name, value in weights.items():
            params[lookup.get(name, name)] = Yahtzotron._translate_parameters({'b': 'bias', 'w': 'kernel'},
                                                                              value)
        return params

    def __repr__(self):
        return f"{self.__class__.__name__}(ruleset={self._ruleset}, objective={self._objective})"

    def create_eager_lut(self):
        self._ruleset.set_lut(create_lut(self._roll_lut_path, self._ruleset))
