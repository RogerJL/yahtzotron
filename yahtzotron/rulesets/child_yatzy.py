from .base import make_category, Ruleset


@make_category
def ones(roll, filled_categories, scores):
    return roll[0]


@make_category
def twos(roll, filled_categories, scores):
    return roll[1]


@make_category
def threes(roll, filled_categories, scores):
    return roll[2]


@make_category
def fours(roll, filled_categories, scores):
    return roll[3]


@make_category
def fives(roll, filled_categories, scores):
    return roll[4]


@make_category
def sixes(roll, filled_categories, scores):
    return roll[5]


@make_category
def one_pair(roll, filled_categories, scores):
    best_pair = 0
    for die_value, num_dice in enumerate(roll, 1):
        if num_dice >= 2:
            best_pair = die_value

    return 4 if best_pair else 0


@make_category
def two_pairs(roll, filled_categories, scores):
    best_pairs = [0, 0]
    for die_value, num_dice in enumerate(roll, 1):
        if num_dice >= 2:
            best_pairs[1] = best_pairs[0]
            best_pairs[0] = die_value

    if best_pairs[1] == 0:
        return 0

    return 6


@make_category
def three_of_a_kind(roll, filled_categories, scores):
    for die_value, num_dice in enumerate(roll, 1):
        if num_dice >= 3:
            return 8

    return 0


@make_category
def four_of_a_kind(roll, filled_categories, scores):
    for die_value, num_dice in enumerate(roll, 1):
        if num_dice >= 4:
            return 10

    return 0


@make_category
def all_differs(roll, filled_categories, scores):
    if all(roll <= 1):
        return 12

    return 0

@make_category
def full_house(roll, filled_categories, scores):
    doublet = None
    triplet = None

    for die_value, num_dice in enumerate(roll, 1):
        if num_dice == 2:
            doublet = die_value
        elif num_dice == 3:
            triplet = die_value

    if doublet is None or triplet is None:
        return 0

    return 14


@make_category
def yatzy(roll, filled_categories, scores):
    if 5 in roll:
        return 20

    return 0


child_yatzy_rules = Ruleset(
    ruleset_name="child_yatzy",
    num_dice=5,
    categories=(
        ones,
        twos,
        threes,
        fours,
        fives,
        sixes,
        one_pair,
        two_pairs,
        three_of_a_kind,
        four_of_a_kind,
        full_house,
        all_differs,
        yatzy,
    ),
    bonus_cutoff=6 * 5 + 1,
    bonus_score=0,
)
