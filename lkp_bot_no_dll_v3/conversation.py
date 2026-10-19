"""Bounded dialogue context: user clarifications, topic changes and expiry."""
from __future__ import annotations
import re
import time
from search_engine import query_terms

HISTORY_TTL = 30 * 60
MAX_TURNS = 4


def relevant_history(question: str, turns: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if not turns:
        return []
    terms = set(query_terms(question))
    old = set(query_terms(' '.join(q for q, _ in turns[-2:])))
    continuation = bool(re.match(r'(?i)^\s*(?:а\s+(?:если|когда|как|можно|без|с|сколько)|тогда|а это|а так|почему так|он\b|она\b|оно\b|это\b|тот\b|этот\b|сколько\s+(?:ждать|сушить)|и\s+(?:как|что)|то есть|да[, ]|нет[, ])', question))
    pending_answer = turns[-1][1].rstrip().endswith('?')
    explicit_question = bool(re.match(r'(?i)^\s*(?:как|почему|чем|зачем|можно|что|сколько|какой|какая)\b', question))
    clarification = pending_answer and len(question) < 180 and '?' not in question and not explicit_question
    # A full explicit question starts its own topic, even when it is short.
    if clarification or continuation or (len(terms) <= 2 and terms & old):
        return turns[-MAX_TURNS:]
    return []


def search_question(question: str, turns: list[tuple[str, str]]) -> str:
    relevant = relevant_history(question, turns)
    if not relevant:
        return question
    # Only user questions can define retrieval. Previous model answers are not
    # evidence and must never feed generated technical claims back into search.
    context = list(dict.fromkeys([relevant[0][0]] + [q for q, _ in relevant[-2:]]))
    return ' '.join(context) + ' ' + question


def load_turns(data: dict, now: float | None = None) -> list[tuple[str, str]]:
    now = time.time() if now is None else now
    if now - data.get('turns_at', 0) > HISTORY_TTL:
        data.pop('turns', None)
        return []
    return data.get('turns', [])[-MAX_TURNS:]


def remember(data: dict, question: str, answer: str, turns: list[tuple[str, str]]) -> None:
    updated = relevant_history(question, turns) + [(question, answer)]
    data['turns'] = updated if len(updated) <= MAX_TURNS else updated[:1] + updated[-(MAX_TURNS-1):]
    data['turns_at'] = time.time()
