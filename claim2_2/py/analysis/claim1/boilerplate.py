"""Boilerplate filter for WGOITG comment sentences.

Replaces the four unanchored regexes carried over from
`visual-inference-analysis/2_sentence_similarity_filtering.ipynb`. Those matched
anywhere in the sentence, so "This relates to my community because I am a high
school athlete." was thrown away along with the bare "This relates to my
community."

The heuristic here is *residual content*, not surface form:

  1. Normalise — strip enumerators ("1.)"), quotes, whitespace.
  2. Layer 1: exact scaffolding. A handful of whole-sentence patterns that are
     pure prompt furniture ("The headline is:", a lone enumerator).
  3. Layer 2: content test. Tokenise, discard function words, prompt verbs,
     words naming the artifact itself, and contentless placeholders. If *any*
     token survives, the reader said something and the sentence is kept.

So the stock opener no longer decides anything. What follows it does.
"""
from __future__ import annotations

import re

# --- 1. normalisation -------------------------------------------------------

_ENUMERATOR = re.compile(r"^\s*\(?\d+\s*[.)\]]+\s*")
_QUOTES = '"“”‘’\''


def normalize(text: str) -> str:
    t = (text or "").replace("’", "'").strip()
    t = _ENUMERATOR.sub("", t)
    return t.strip().strip(_QUOTES).strip()


# --- 2. exact scaffolding ---------------------------------------------------

_SCAFFOLD = [
    r"\d+[.)\]]*",                                              # a lone enumerator
    r"(?:my |our )?(?:catchy |possible |good )?(?:headline|title)"
    r"(?: would be| could be| is| for this(?: graph)? is)?:?",  # headline stubs
    r"the (?:headline|title)(?: would be| is)?:?",
    r"the impact of the (?:graph|chart|data)s? on (?:my|our) community is(?: that)?",
    r"(?:step|part|question|number) \d+[.:]?",
]
_SCAFFOLD_RE = re.compile(r"^(?:%s)[.!?]?$" % "|".join(_SCAFFOLD), re.IGNORECASE)


# --- 3. content test --------------------------------------------------------

# Function words: pronouns, determiners, auxiliaries, prepositions, conjunctions.
_FUNCTION = """
a an the this that these those there here it its it's his her their our my your
i me we us you he she they them him myself ourselves themselves
is are was were be been being am do does did done doing have has had having
will would can could shall should may might must
of in on at to for from by with about into over under between among across
and or but so because since as if then than when where while whether that
not no nor none never only just also too very quite rather really actually
what which who whom whose why how
""".split()

# The artifact and the exercise itself — naming them is not saying anything.
_ARTIFACT = """
graph graphs chart charts plot plots map maps image images picture pictures
figure figures diagram visual visuals article headline title caption data
dataset graphic infographic
""".split()

# Prompt verbs and other verbs that carry no claim on their own.
_WEAK_VERBS = """
notice noticed notices see saw seen look looked looks watch watched wonder
wondered wonders ask asked asks think thought thinks consider considered
know knew say says said tell shows show showed shown display displays
relate relates related relating apply applies affect affects impact impacts
exist exists existed happen happens happened mean means concern concerns
involve involves include includes get got go goes come comes make makes take
takes give gives put puts find found seem seems appear appears
""".split()

# Placeholders and social deixis — present in the sentence, absent in meaning.
_PLACEHOLDER = """
something anything nothing everything thing things stuff lot lots bit
some any many much few several more less most least other others another
same different difference differences big small large little huge tiny
good bad great nice interesting important cool weird strange
community communities people person persons everyone someone anyone folks
scenario situation case example question questions answer answers idea ideas
way ways kind sort type point points part parts first second last next
directly indirectly closely together well overall general generally
one two both all each every
""".split()

_STOPWORDS = frozenset(_FUNCTION + _ARTIFACT + _WEAK_VERBS + _PLACEHOLDER)

# Dotted acronyms (U.S., U.K.) first, so they survive as one token.
_TOKEN = re.compile(r"[A-Za-z](?:\.[A-Za-z])+\.?|[A-Za-z][A-Za-z'-]*|\d[\d,.%]*")


def content_tokens(text: str) -> list[str]:
    """Tokens that carry meaning: not function words, artifact names, prompt
    verbs or placeholders. Numbers always count."""
    out: list[str] = []
    for i, tok in enumerate(_TOKEN.findall(normalize(text))):
        # A capitalised token after the first word is a proper noun — a place, a
        # person, a series label. It is content whatever the stoplists say.
        bare = tok.replace(".", "")
        is_acronym = len(bare) >= 2 and bare.isupper()
        if is_acronym or (
            i > 0 and tok[:1].isupper() and len(tok) >= 2 and tok.lower() not in _STOPWORDS
        ):
            out.append(tok.lower())
            continue
        low = tok.lower().strip("'-")
        if low.endswith("'s"):  # possessives: data's -> data
            low = low[:-2]
        if not low:
            continue
        if low[0].isdigit():
            out.append(low)
            continue
        if low in _STOPWORDS:
            continue
        if len(low) < 3:  # stray initials, "a", "u"
            continue
        out.append(low)
    return out


# A sentence long enough to carry a clause is not prompt furniture, even when
# every word of it is on the weak lists ("I wonder if chart A and chart B graph
# the same data in a different way."). Only short frames can be scaffolding.
MAX_SCAFFOLD_WORDS = 12


def is_boilerplate(text: str, max_words: int = MAX_SCAFFOLD_WORDS) -> bool:
    """True when the sentence is prompt scaffolding, an echo of the question,
    or a short frame with nothing inside it."""
    t = normalize(text)
    if not t or not any(c.isalpha() or c.isdigit() for c in t):
        return True
    if _SCAFFOLD_RE.match(t):
        return True
    if len(_TOKEN.findall(t)) > max_words:
        return False
    return not content_tokens(t)


__all__ = ["is_boilerplate", "content_tokens", "normalize"]
