
from dataclasses import dataclass
from typing import TypeAlias
from nltk import pos_tag, word_tokenize
from functools import cached_property

def simple_pos(tag: str) -> str:
    """Convert a Penn Treebank tag to its first two characters."""
    return tag[:2]


@dataclass(frozen=True)
class PredicateToken:
    word: str
    tag: str

    @cached_property
    def simple_tag(self) -> str:
        return self.tag[:2]

@dataclass(frozen=True)
class Predicate:
    tokens: tuple[PredicateToken, ...]

    @classmethod
    def from_string(cls, text: str) -> 'Predicate':

        words =  word_tokenize(text)
        tagged_words = pos_tag(words)

        return cls(tuple(
            PredicateToken(
                word=word,
                tag=tag,
            )
            for word, tag in tagged_words
        ))

    @property
    def tag(self) -> str:
        return " ".join(token.tag for token in self.tokens)

    @property
    def simple_tag(self) -> str:
        return " ".join(token.simple_tag for token in self.tokens)

    @property
    def words(self) -> tuple[str, ...]:
        return tuple(token.word for token in self.tokens)

    @property
    def tags(self) -> tuple[str, ...]:
        return tuple(token.tag for token in self.tokens)

    @property
    def simple_tags(self) -> tuple[str, ...]:
        return tuple(token.simple_tag for token in self.tokens)

    @property
    def predicate(self) -> str:
        return ' '.join([token.word for token in self.tokens])

    def __str__(self) -> str:
        return " ".join(self.words)


# ── LHS: pattern elements ─────────────────────────────────

class Pattern:
    """Base class for LHS pattern elements."""

@dataclass(frozen=True)
class Token(Pattern):
    values: frozenset[str]

    def __init__(self, values: str | list[str]):

        if isinstance(values, str):
            values = [values]
        
        object.__setattr__(self, "values", frozenset(values))


@dataclass(frozen=True)
class Tag(Pattern):
    value: str

    @property
    def simple_tag(self) -> str:
        return simple_pos(self.value)


@dataclass(frozen=True)
class Placeholder(Pattern):

    size: int | str

    def __post_init__(self) -> None:

        if self.size not in ["*", "+"] and (
            not isinstance(self.size, int) or self.size < 0
        ):
            raise ValueError("size must be a non-negative integer or '*'.")

# ── RHS: transformation operations ─────────────────────────

class Operation:
    """Base class for RHS operations."""


@dataclass(frozen=True)
class _KEEP(Operation):
    """Keep every token matched by the corresponding LHS element."""


@dataclass(frozen=True)
class _DELETE(Operation):
    """Delete every token matched by the corresponding LHS element."""

KEEP = _KEEP()
DELETE = _DELETE()


@dataclass(frozen=True)
class REPLACE(Operation):
    """Replace a matched span with a string or a captured token."""

    value: str | int


# ── Rule ───────────────────────────────────────────────────

PatternList: TypeAlias = list[Pattern]
OperationList: TypeAlias = list[Operation]
Span: TypeAlias = tuple[int, int]
Captures: TypeAlias = list[Span]
Match: TypeAlias = tuple[int, int, Captures]


class Rule:

    def __init__(
        self,
        lhs: list[Pattern],
        rhs: list[Operation],
    ) -> None:
        if len(lhs) != len(rhs):
            raise ValueError("LHS and RHS must have the same length.")

        self.lhs = lhs
        self.rhs = rhs

    def _match(
        self,
        predicate: Predicate,
        start: int = 0,
        full: bool = True,
        greedy: bool = True,
    ) -> Captures | None:
        
        tokens = predicate.tokens
        n = len(tokens)
        captures: Captures = []
        result: Captures | None = None

        def search(i: int, j: int) -> bool:

            nonlocal result

            if j == len(self.lhs):
                if not full or i == n:
                    result = captures.copy()
                    return True
                return False

            part = self.lhs[j]
            lengths: list[int] = []

            if isinstance(part, (Token, Tag)):
                if i < n:
                    if isinstance(part, Token):
                        valid = tokens[i].word in part.values
                    else:
                        valid = (
                            tokens[i].tag == part.value
                            or tokens[i].simple_tag == part.simple_tag
                        )

                    if valid:
                        lengths = [1]

            elif isinstance(part, Placeholder):
                if isinstance(part.size, int):
                    if i + part.size <= n:
                        lengths = [part.size]

                elif part.size in ("*", "+"):
                    minimum = 0 if part.size == "*" else 1
                    lengths = list(range(minimum, n - i + 1))

                    if greedy:
                        lengths.reverse()

                else:
                    raise ValueError(
                        f"Unsupported placeholder size: {part.size!r}"
                    )

            else:
                raise TypeError(f"Unsupported pattern: {part!r}")

            matched = False

            for length in lengths:
                captures.append((i, i + length))

                if search(i + length, j + 1):
                    matched = True
                    break

                captures.pop()

            return matched

        search(start, 0)
        return result

    def matches(self, predicate: Predicate) -> bool:
        captures = self._match(predicate, full=True)
        return captures is not None

    def _find_first_match(
        self,
        predicate: Predicate,
    ) -> Match | None:
        n = len(predicate.tokens)
        result: Match | None = None

        for start in range(n + 1):
            captures = self._match(
                predicate,
                start=start,
                full=False,
                greedy=False,
            )

            if captures is not None:
                end = captures[-1][1] if captures else start
                result = (start, end, captures)
                break

        return result

    def find_first(
        self,
        predicate: Predicate,
    ) -> Span | None:
        
        match = self._find_first_match(predicate)

        if match is None:
            return None

        start, end, captures = match

        # Return the span excluding leading/trailing context placeholders.
        first = 0
        last = len(captures) - 1

        while (
            first < len(self.lhs)
            and isinstance(self.lhs[first], Placeholder)
            and self.lhs[first].size in ("*", "+")
        ):
            first += 1

        while (
            last >= first
            and isinstance(self.lhs[last], Placeholder)
            and self.lhs[last].size in ("*", "+")
        ):
            last -= 1

        if first > last:
            return start, end

        return captures[first][0], captures[last][1]

    def _transform(
        self,
        predicate: Predicate,
        captures: Captures,
    ) -> list[str]:
        
        output: list[str] = []

        for operation, (start, end) in zip(self.rhs, captures):

            if isinstance(operation, _KEEP):
                output.extend(token.word for token in predicate.tokens[start:end])

            elif isinstance(operation, _DELETE):
                pass

            elif isinstance(operation, REPLACE):
                if isinstance(operation.value, str):
                    output.append(operation.value)
                else:
                    output.append(
                        predicate.tokens[operation.value].word
                    )

            else:
                raise TypeError(
                    f"Unsupported operation: {operation!r}"
                )

        return output

    def apply(self, predicate: Predicate) -> str:

        result = predicate.predicate

        if self.matches(predicate):

            captures = self._match(predicate, full=True)

            if captures is not None:
                result = " ".join(self._transform(predicate, captures))

        return result

    def apply_first(self, string : str) -> str:

        predicate = Predicate.from_string(string)
        match = self._find_first_match(predicate)
        result = string

        if match is not None:

            start, end, captures = match
            transformed = self._transform(predicate, captures)

            result = " ".join([
                *(token.word for token in predicate.tokens[:start]),
                *transformed,
                *(token.word for token in predicate.tokens[end:]),
            ])

        return result


def parse_lhs(lhs: str) -> list[Pattern]:

    result: list[Pattern] = []

    for part in lhs.split():
        if part.startswith("{") and part.endswith("}"):
            result.append(Token(part[1:-1].split("|")))

        elif part.startswith("[") and part.endswith("]"):
            result.append(Tag(part[1:-1]))

        elif part in ("*", "+"):
            result.append(Placeholder(part))

        elif part.startswith("*") and part[1:].isdigit():
            result.append(Placeholder(int(part[1:])))

        else:
            result.append(Token(part))

    return result


def parse_rhs(rhs: str) -> list[Operation]:

    result: list[Operation] = []

    for part in rhs.split():
        if part == "=":
            result.append(KEEP)

        elif part == "~":
            result.append(DELETE)

        elif part.startswith(">@"):
            result.append(REPLACE(int(part[2:])))

        elif part.startswith(">"):
            result.append(REPLACE(part[1:]))

        else:
            raise ValueError(f"Unknown RHS operation: {part}")

    return result


def parse_rule(lhs: str, rhs: str) -> Rule:
    return Rule(parse_lhs(lhs), parse_rhs(rhs))