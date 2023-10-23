from .yatzy import yatzy_rules
from .yahtzee import yahtzee_rules
from .child_yatzy import child_yatzy_rules

AVAILABLE_RULESETS = {r.name: r for r in (yatzy_rules, yahtzee_rules, child_yatzy_rules)}
