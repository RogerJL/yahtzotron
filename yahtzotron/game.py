import numpy as np
from jax.random import PRNGKey


class Scorecard:
    """Keep track of scores during the game."""

    def __init__(self, ruleset, scores=None):
        self.scores = np.zeros(ruleset.num_categories, dtype="int")
        self.filled = np.zeros(ruleset.num_categories, dtype="bool")

        if scores is not None:
            assert len(scores) == ruleset.num_categories
            for i in range(ruleset.num_categories):
                if scores[i] is not None:
                    self.scores[i] = scores[i]
                    self.filled[i] = 1

        self.ruleset_ = ruleset

    def copy(self):
        new_card = Scorecard(self.ruleset_)
        new_card.scores[...] = self.scores
        new_card.filled[...] = self.filled
        return new_card

    def register_score(self, roll, cat_index):
        roll = np.array(roll)
        total_score_old = self.total_score()
        score = self.ruleset_.score(roll, cat_index, self.filled, self.scores)
        self.scores[cat_index] = score
        self.filled[cat_index] = 1
        return self.total_score() - total_score_old

    def score_summary(self):
        return np.array(self.ruleset_.score_summary(self.scores))

    def total_score(self) -> int:
        return self.ruleset_.total_score(self.scores)

    def to_array(self):
        return np.concatenate([self.filled, self.score_summary()])

    def __repr__(self):
        score_str = ", ".join(
            str(score) if filled else "None"
            for score, filled in zip(self.scores, self.filled)
        )
        return (
            f"{self.__class__.__name__}(ruleset={self.ruleset_}, scores=[{score_str}])"
        )


def play_tournament(agents, deterministic_rolls=False, record_trajectories=False, rngs: PRNGKey = None):
    """Play a tournament between given agents.

    Returns either final scores or final scores and all trajectories
    (if record_trajectories=True). A trajectory is a sequence of
    (observation, action, reward) tuples for each agent.
    """
    if not hasattr(agents, "__iter__"):
        agents = [agents]

    num_players = len(agents)

    ruleset = agents[0].ruleset()
    for a in agents:
        if a.ruleset() != ruleset:
            raise ValueError("Only agents sharing the same ruleset can play each other")

    num_dice = ruleset.num_dice

    scores = [Scorecard(ruleset) for _ in range(num_players)]

    # initialize even when not used
    trajectories = [[] for _ in range(num_players)]

    rngs1 = np.random.default_rng(rngs.tolist())
    for _ in range(ruleset.num_rounds):
        if deterministic_rolls:
            player_rolls = (np.tile(

                rngs1.integers(size=(1, 3, num_dice), low=1, high=7),
                (num_players, 1, 1))
            )
        else:
            player_rolls = rngs1.integers(size=(num_players, 3, num_dice), low=1, high=7)

        for p in range(num_players):
            my_score = scores[p]
            other_scores = [scores[i] for i in range(num_players) if i != p]

            turn_iter = agents[p].turn(my_score, other_scores)
            for roll_num, kept_dice in enumerate(turn_iter):
                num_to_roll = num_dice - len(kept_dice)
                dice_ = np.asarray((0,) * num_to_roll + kept_dice)
                new_roll = np.where(dice_ != 0, dice_, player_rolls[p, roll_num])
                turn_state = turn_iter.send(new_roll)

                if turn_state["rolls_left"] == 0:
                    reward = scores[p].register_score(
                        turn_state["dice_count"], turn_state["category_idx"]
                    )
                else:
                    reward = 0

                if record_trajectories:
                    trajectories[p].append(
                        (
                            turn_state["observation"],
                            turn_state["action"],
                            reward,
                        )
                    )

    return scores, trajectories
