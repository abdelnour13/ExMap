from typing import List, Tuple, Any, Literal, Dict
from src.commands.knowledge_graph_construction.graph_cleaning.utils import (
    Placeholder,
    Rule,
    Token,
    Tag,
    DELETE,
    KEEP,
    REPLACE,
    Predicate,
    parse_rule,
    PredicateToken
)
from src.helpers import load_json_file
from pydantic import BaseModel

VAR_NAMES = Literal[
    'white_tokens',
    'entities_stopwords',
    'predicates_stopwords',
    'propositions'
]

class RulesFile(BaseModel):
    variables : Dict[VAR_NAMES, List[str]]

# --------------------- Predefined tokens -----------------------------

TOKENS_TO_REMOVE = [
    "a", 
    "the", 
    "an", 
    "also", 
    "key", 
    "new",
    "us"
]

STOP_ENTITIES = [
    "interest",
    "researcher",
    "researchers",
    "phd",
    "thesis",
    "study",
    "studies",
    "approach",
    "work",
    "article",
    "chapter",
    "section",
    "author",
    "results",
    "experiment",
    "experiments",
    "objective",
    "reasearch",
    "conference",
    "contribution",
    "survey",
    "paper",
    "experimental",
    "comparision",
    "we",
    "it",
    "research",
    "literature",
    "review",
    "make",
    "our",
    "proposed",
    "result",
    "supplementary"
]

STOP_PREDICATES = [
    "suggest",
    "introduce",
    "show",
    "present",
    "can",
    "cannot",
    "will",
    "could",
    "like",
    "hard",
    "impossible",
    "not",
    "n't",
    "unable",
    "difficult",
    "rarely",
    "negatively",
    "negative",
    "fail",
    "no",
    "became",
    "should",
    "must",
    "may",
    "without",
    "est",
    "non",
    "incomplete",
    "lack",
    "unavailable",
    "investigate", 
    "investigates",
    "inability",
    "need",
    "needs",
    "insights",
    "insight",
    "attention",
    "poorly"
]

PROPOSITIONS = [
    "of",
    "on",
    "in",
    "with",
    "into",
    "onto",
    "between",
    "below",
    "above",
    "under",
    "by",
    "for",
    "at",
    "over",
    "throught",
    "have",
    "to",
    "from"
]

INVALID_PATTERNS = [
    "IN",
    "TO",
    "JJ NN IN",
    "NN IN NN IN",
    'JJ IN',
    'MD VB',
    'VB RB',
    'JJ NN',
    'NN RB',
    'RB TO',
    'NN JJ',
    'IN NN',
    'IN IN',
    'NN WR',
    'VB WR',
    'JJ VB',
    'RB JJ',
    'WR VB',
    'NN NN CD NN',
    'VB DT',
    'IN JJ',
    'FW NN',
    'JJ JJ'
]

USED_TO_VARIANTS = Token([
    "used",
    "employed",
    "designed",
    "intended",
    "aimed",
    "helped",
    "allowed",
    "enabled",
    "required",
    "needed",
    "attempted",
    "sought",
    "tended",
    "failed",
    "expected",
    "chosen",
    "selected",
    "adopted",
    "applied",
    "proposed",
    "developed",
    "introduced", 
    "trained", 
    "optimized", 
    "presented", 
    "conducted", 
    "adapted"
])

USED_AS_VARIANTS = Token([
    "used",
    "serves",
    "employed"
])

AIMS_TO_VARIANTS = [
    'aims',
    'allows',
    'enables',
    'leads',
    'proposes',
    'serves',
    'contributes',
    'tends',
    'permits',
    'seeks',
    'learns',
    'attempts',
    'helps',
    'hopes',
    'offers',
    'sizes',
    'imposes',
    'extends',
    'forces',
    'opts',    
]

AIMS_TO_VARIANTS = AIMS_TO_VARIANTS + [v.rstrip('s') for v in AIMS_TO_VARIANTS] + [f'{v.rstrip('s')}ing' for v in AIMS_TO_VARIANTS]
AIMS_TO_VARIANTS = Token(AIMS_TO_VARIANTS)

HAS_VARIANTS = Token([
    "has",
    "have",
    "had"
])

DESIGNED_TO_VARIANTS = Token([
    'designed',
    'used',
    'proposed',
    'led',
    'introduced',
    'expected',
    'extended',
    'integrated',
    'engineered',
    'implemented',
    'developed',
    'trained',
    'intended',
    'enhanced',
    'employed',
    'modified',
    'required',
    'adapted',
    'corrected',
    'found',
    'demonstrated',
    'conducted',
    'proved',
    'updated',
    'guaranteed',
    'allowed',
    'tuned',
])

IS_VARIANTS = Token(['is', 'are', 'were', 'was', "be", "been"])

GOOD_FOR_VARIANTS = Token(['capable',
    'essential',
    'effective',
    'useful',
    'suitable',
    'responsible',
    'significant',
    'necessary',
    'potential',
    'important',
    'utilized',
    'accessible',
    'good'
])

IMPORTANT_VARIANTS = Token([
    "important",
    "critical",
    "crucial",
    "significant",
    "pivotal",
    "fundamental",
    "essential",
    'appropriate',
    'sufficient',
] + list(GOOD_FOR_VARIANTS.values))

ABILITY = Token([
    'potential',
    'ability',
    'capacity',
    'possibility',
    'opportunity',
])

PLAYS_ROLE_IN =  [
    [Token("plays"), Token("role"),],
    [Token("play"), Token("role"),],
    [Token("holds"), Token("potential"),],
    [Token("hold"), Token("potential"),],
    [Token("needed")],
    [Token("is needed")],
    [IS_VARIANTS, IMPORTANT_VARIANTS],
    [IMPORTANT_VARIANTS]
]

REMAIN_VARIANTS = Token([
    "remains"
])

HEAD = Token("{head}")
TAIL = Token("{tail}")
SEP = Token("|")

#------------------------ Pre-defined equivalent sets --------------------------------------

EQUIV_SETS : List[List[Tuple[str, bool]]] = [
    [
        ("essential for", False),
        ("needed for", False),
        ("is needed for", False),
        ("needs", True)
    ],
    [ 
        ("use", False),
        ("make use of", False)
    ],
    [
        ("is equivelant to", False), 
        ("also known as", False), 
        ("is also known as", False), 
        ("is known as", False), 
        ("known as", False), 
        ("same as", False), 
        ("is same as", False)
    ],
    [
        ("have abbreviation", False), 
        ("is namely", False), 
        ("is abbreviated as", False), 
        ("are abbreviated as", False), 
        ("is abbreviation of", True), 
        ("is referred to as", False), 
        ("is refered to", False), 
        ("referred to as", False)
    ],
    [
        ("causes", False),
        ("cause", False), 
        ("because of", True), 
        ("because", True), 
        ("is because of", True),
        ("are because of", True), 
        ("was because of", True),
        ("were because of", True),
        ("caused", False), 
        ("due to", True), 
        ("due", True), 
        ("is due", True), 
        ("is due to", True), 
        ("are due", True), 
        ("are due to", True), 
        ("was due", True), 
        ("was due to", True), 
        ("were due", True), 
        ("were due to", True), 
        ("lead to", False),
        ("led to", False),
        ("leads to", False)
    ]
]

# --------------------------------- Rules --------------------------------------

def select(cond : bool, arg0 : Any, arg1 : Any) -> Any:
    return [arg0, arg1][int(cond)]

RULES = [

    *[
        Rule(
            lhs=[
                HEAD,
                SEP,
                *[Token(token) for token in predicate.split(' ')], 
                SEP,
                TAIL
            ],
            rhs=[
                REPLACE(select(flip, -1, 0)),
                KEEP,
                REPLACE(set_[0][0]), 
                *[DELETE for _ in range(len(predicate.split(' ')) - 1)],
                KEEP,
                REPLACE(select(flip, 0, -1))
            ]
        )
        for set_ in EQUIV_SETS
        for predicate, flip in set_
    ],

    parse_rule(
        lhs="* [VB] [CC] [VB] *",
        rhs="= ~ ~ = ="
    ),

    Rule(
        lhs=[HEAD, SEP, Tag("VB"), Token(["by", "using"]), SEP, TAIL],
        rhs=[REPLACE(-1), KEEP, KEEP, DELETE, KEEP, REPLACE(0)]
    ),

    Rule(
        lhs=[HEAD, SEP, Tag("VB"), Token("by"), Token("using"), SEP, TAIL],
        rhs=[REPLACE(-1), KEEP, KEEP, DELETE, DELETE, KEEP, REPLACE(0)]
    ),

    Rule(
        lhs=[HEAD, SEP, Tag("RB"), Tag("VB"), Token("by"), SEP, TAIL],
        rhs=[REPLACE(-1), KEEP, REPLACE(2), REPLACE(1), DELETE, KEEP, REPLACE(0)]
    ),

    Rule(
        lhs=[HEAD, SEP, Tag("VB"), Tag("VB"), Token("by"), SEP, TAIL],
        rhs=[REPLACE(-1), KEEP, DELETE, KEEP, DELETE, KEEP, REPLACE(0)]
    ),

    Rule(
        lhs=[HEAD, SEP, Tag("NN"), Tag("VB"), Token("by"), SEP, TAIL],
        rhs=[REPLACE(-1), KEEP, REPLACE(2), REPLACE(1), DELETE, KEEP, REPLACE(0)]
    ),

    Rule(
        lhs=[HEAD, SEP, Tag("MD"), Placeholder("+"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, KEEP, KEEP, KEEP]
    ),  # can be used to => be used to

    Rule(
        lhs=[HEAD, SEP, REMAIN_VARIANTS, Placeholder("+"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, KEEP, KEEP, KEEP]
    ),  # remains important in training => important in training

    Rule(
        lhs=[HEAD, SEP, Token("be"), Placeholder("+"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, KEEP, KEEP, KEEP]
    ),  # be used to => used to

    Rule(
        lhs=[HEAD, SEP, HAS_VARIANTS, Token("been"), Placeholder("+"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, DELETE, KEEP, KEEP, KEEP]
    ),  # has been used to => used to

    Rule(
        lhs=[HEAD, SEP, USED_AS_VARIANTS, Token("as"), Tag("NN"), Tag("IN"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, DELETE, KEEP, KEEP, KEEP, KEEP]
    ),  # used as input for => input for

    Rule(
        lhs=[HEAD, SEP, DESIGNED_TO_VARIANTS, Tag("TO"), Tag("VB"), Tag("IN"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, DELETE, KEEP, KEEP, KEEP, KEEP]
    ),  # designed to deal with => deal with

    Rule(
        lhs=[HEAD, SEP, IS_VARIANTS, Tag("JJ"), Tag("NNS"), Tag("IN"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, KEEP, KEEP, KEEP, KEEP, KEEP]
    ),  # are core components of => core components of

    Rule(
        lhs=[HEAD, SEP, AIMS_TO_VARIANTS, Tag("TO"), Tag("VB"), Tag("IN"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, DELETE, KEEP, KEEP, KEEP, KEEP]
    ),  # aims to learn about => learn

    *[
        Rule(
            lhs=[HEAD, SEP, *variant, Tag("IN"), Tag("VB"), SEP, TAIL],
            rhs=[KEEP, KEEP, *[DELETE for _ in variant], DELETE, KEEP, KEEP, KEEP]
        )
        for variant in PLAYS_ROLE_IN
    ],  # plays role in enhancing => enhancing

    Rule(
        lhs=[HEAD, SEP, HAS_VARIANTS, ABILITY, Tag("TO"), Tag("VB"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, DELETE, DELETE, KEEP, KEEP, KEEP]
    ),  # aims to reduce => reduce

    Rule(
        lhs=[HEAD, SEP, USED_TO_VARIANTS, Tag("TO"), Tag("VB"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, DELETE, KEEP, KEEP, KEEP]
    ),  # used to evaluate => evaluate

    Rule(
        lhs=[HEAD, SEP, IS_VARIANTS, USED_TO_VARIANTS, Tag("TO"), Tag("VB"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, DELETE, DELETE, KEEP, KEEP, KEEP]
    ),  # is used to evaluate => evaluate

    Rule(
        lhs=[HEAD, SEP, IS_VARIANTS, Tag("NN"), Tag("IN"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, KEEP, KEEP, KEEP, KEEP]
    ),  # is part of => part of

    Rule(
        lhs=[HEAD, SEP, IS_VARIANTS, Tag("VBN"), Tag("IN"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, KEEP, KEEP, KEEP, KEEP]
    ),  # is used in => used in

    Rule(
        lhs=[HEAD, SEP, IS_VARIANTS, Tag("JJ"), Tag("IN"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, KEEP, KEEP, KEEP, KEEP]
    ),  # is crucial for => crucial for

    Rule(
        lhs=[HEAD, SEP, IS_VARIANTS, Tag("JJ"), Tag("TO"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, KEEP, KEEP, KEEP, KEEP]
    ),  # is crucial for => crucial for

    Rule(
        lhs=[HEAD, SEP, AIMS_TO_VARIANTS, Tag("TO"), Tag("VB"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, DELETE, KEEP, KEEP, KEEP]
    ),  # aims to reduce => reduce

    Rule(
        lhs=[HEAD, SEP, AIMS_TO_VARIANTS, Tag("IN"), Tag("VBG"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, DELETE, KEEP, KEEP, KEEP]
    ),  # aims at reducing => reducing

    Rule(
        lhs=[HEAD, SEP, Token("more"), Tag("JJ"), Tag("IN"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, KEEP, KEEP, KEEP, KEEP]
    ),  # more effective for => effective for

    Rule(
        lhs=[HEAD, SEP, IS_VARIANTS, Token("more"), Tag("JJ"), Tag("IN"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, DELETE, KEEP, KEEP, KEEP, KEEP]
    ),  # is more effective for => effective for

    Rule(
        lhs=[HEAD, SEP, Token("used"), Token("for"), Tag("NN"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, DELETE, KEEP, KEEP, KEEP]
    ),  # used for training => train

    Rule(
        lhs=[HEAD, SEP, GOOD_FOR_VARIANTS, Tag("IN"), Tag("VBG"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, DELETE, KEEP, KEEP, KEEP]
    ),  # good in reducing => reduces

    Rule(
        lhs=[HEAD, SEP, IS_VARIANTS, Tag("VBG"), Tag("IN"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, KEEP, KEEP, KEEP, KEEP]
    ),  # is challenging for => challenging for

    Rule(
        lhs=[HEAD, SEP, IS_VARIANTS, Tag("CD"), Tag("IN"), SEP, TAIL],
        rhs=[KEEP, KEEP, DELETE, KEEP, KEEP, KEEP, KEEP]
    ),  # is one of
]

def apply_all_rules(predicate : str, rules : list[Rule]) -> str:

    result = "{head} | {predicate} | {tail}"

    for rule in rules:

        predicate = Predicate.from_string(predicate)

        predicate = Predicate(tuple([
            PredicateToken(next(iter(HEAD.values)), "NN"),
            PredicateToken(next(iter(SEP.values)), "NN"),  
            *predicate.tokens, 
            PredicateToken(next(iter(SEP.values)), "NN"), 
            PredicateToken(next(iter(TAIL.values)), "NN"), 
        ]))

        predicate = rule.apply(predicate)
        head, predicate, tail = predicate.split(' | ')
        result = result.format(head=head, tail=tail, predicate="{predicate}")

    result = result.format(predicate=predicate, head="{head}", tail="{tail}")

    return result

if __name__ == '__main__':
    print(apply_all_rules(" ", RULES))