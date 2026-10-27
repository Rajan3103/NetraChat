"""
Grammar helpers that keep every reply lexically and grammatically correct.

Problems these helpers solve:
  * articles      -> "a umbrella"   becomes "an umbrella"
  * agreement     -> "there is 3 person" becomes "there are three people"
  * lists         -> ["a cat", "a dog", "a TV"] becomes "a cat, a dog, and a TV"
  * finishing     -> capital letter at the start, one full stop at the end,
                     no double spaces, no repeated words ("the the").
"""

from __future__ import annotations

import re

import inflect

from ..vocabulary import plural, singular

_p = inflect.engine()


def number_word(n: int) -> str:
    """3 -> 'three'. Numbers above twenty stay as digits (easier to read)."""
    return _p.number_to_words(n) if 0 <= n <= 20 else str(n)


def with_article(noun_phrase: str) -> str:
    """'umbrella' -> 'an umbrella', 'TV' -> 'a TV', 'hour' -> 'an hour'."""
    return _p.a(noun_phrase)


def count_phrase(label: str, n: int, article: bool = True) -> str:
    """Return a correctly inflected count of a detector class.

    count_phrase('person', 1)  -> 'a person'
    count_phrase('person', 3)  -> 'three people'
    count_phrase('skis', 2)    -> 'two pairs of skis'
    """
    if n == 1:
        return with_article(singular(label)) if article else "one " + singular(label)
    if n == 0:
        return "no " + plural(label)
    return f"{number_word(n)} {plural(label)}"


def be_verb(n: int) -> str:
    """Subject-verb agreement: 'is' for one, 'are' for many / none."""
    return "is" if n == 1 else "are"


def there_be(label: str, n: int) -> str:
    """'there is a dog' / 'there are two dogs' / 'there are no dogs'."""
    return f"there {be_verb(n)} {count_phrase(label, n)}"


def join_list(items: list[str], conj: str = "and") -> str:
    """Join phrases with commas and a final 'and' (Oxford comma)."""
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conj} {items[1]}"
    return ", ".join(items[:-1]) + f", {conj} {items[-1]}"


_ARTICLE_FIX = re.compile(r"\b(a|an|A|An)\s+([A-Za-z][\w-]*)")


def _fix_articles(text: str) -> str:
    def repl(m):
        art, word = m.group(1), m.group(2)
        correct = _p.a(word).split(" ", 1)[0]
        if art[0].isupper():
            correct = correct.capitalize()
        return f"{correct} {word}"
    return _ARTICLE_FIX.sub(repl, text)


def polish(text: str) -> str:
    """Final grammar pass applied to every sentence the bot sends."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\b(\w+) \1\b", r"\1", text, flags=re.IGNORECASE)  # "the the"
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)                       # "dog ."
    text = re.sub(r"([,.!?;:])(?=[A-Za-z])", r"\1 ", text)             # "dog.It"
    text = re.sub(r"\bi\b", "I", text)
    text = _fix_articles(text)
    # Capitalise the first letter of every sentence.
    text = re.sub(r"(^|[.!?]\s+)([a-z])",
                  lambda m: m.group(1) + m.group(2).upper(), text)
    if text and text[-1] not in ".!?" and not re.search(r"[.!?][\"')]$", text):
        text += "."
    return text


def polish_paragraph(sentences: list[str]) -> str:
    return " ".join(polish(s) for s in sentences if s and s.strip())
