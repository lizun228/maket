"""Local FTS retrieval with concept coverage, answer context and overlap removal.

Only reads the knowledge database. Source IDs are retained for developer evaluation,
but never included in context sent to a language model.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import math
from pathlib import Path
import re
import sqlite3

STOPWORDS = set('''и а но или без сколько такой такая такое такие что-то в во на по за с со до от для при из к ко о об про это что как чем где когда если есть нет да вы мы они она он его ее их нас вас мне тебе тут там уже еще только так вообще очень надо нужно можно будет была были был быть почему после перед вопрос здравствуйте подскажите пожалуйста добрый день вечер ребята всем привет делать сделать делаю сделаю делаете сделали какой какое какая какие который которая которые ли же то правильно происходит возникает получается может могут должен должна нужно стал стала стали становится остается остаются проверить пробовал попробовать помогите подскажите объясните причина причины возможно проблема проблемой работать работы лучше совсем сильно именно сразу чуть видна видно виден видны таком этом этом чем чтобы чтоб также материал материала материалы состав состава составом автомобиль машины'''.split())

# Equivalents, not loosely related products/tools. In particular, do not expand
# выдергивать to выдер*: that also matches выдержка, ruining diagnostic searches.
CONCEPTS = {
    'каттер': ('каттер', 'катер', 'cutter', 'подрез', 'подреза', 'среза', 'срезк', 'резак'),
    'отрыв': ('выдира', 'выдера', 'вырыв', 'выдерг', 'вырвал', 'вылет', 'выпад', 'отрыва', 'отвали'),
    'просадка': ('просад', 'усад', 'провал'),
    'сушка': ('суш', 'сохн', 'высох', 'высых', 'просох', 'досох'),
    'кисть': ('кист',),
    'аэрограф': ('аэрограф', 'аэро'),
    'нанесение': ('нанос', 'нанош', 'нанес'),
    'долго': ('долг', 'дольш', 'длитель', 'медлен'),
    'ик': ('ик', 'инфракрас', 'инфракр'),
    'шлифовка': ('шлиф', 'шкур', 'вышкур', 'наждач'),
    'полировка': ('полир', 'заполир', 'отполир'),
    'обезжиривание': ('обезжир', 'антисиликон', 'анти-силикон'),
    'адгезия': ('адгез', 'сцеплен'),
    'цвет': ('цвет', 'оттен', 'колер', 'подбор', 'подобра'),
    'серебро': ('серебр', 'сереб'),
    'металлик': ('металик', 'металлик'),
    'царапина': ('царап',),
    'скол': ('скол',),
    'лак': ('лак',),
    'мокрый': ('мокр', 'водич', 'водой', 'воду', 'воде'),
    'сухой': ('сух', 'сухую'),
    'мягкий': ('мягк', 'резинов', 'липк', 'тянет', 'тяну', 'залипа'),
    'граница': ('границ', 'ореол', 'ареол', 'кант'),
    'разбавитель': ('разбав', 'растворител'),
}
# Compatibility with local developer utilities from earlier releases.
ALIASES = CONCEPTS
TOKEN = re.compile(r'[а-яёa-z0-9]+', re.I)
MESSAGE = re.compile(r'(?m)^№(\d+):\s*')


def normalized(text: str) -> str:
    return text.casefold().replace('ё', 'е')


def _concept(word: str) -> str:
    for name, variants in CONCEPTS.items():
        if any(word == v if len(v) <= 2 else word.startswith(v) for v in variants):
            return name
    return word[:5] if len(word) > 6 else (word[:4] if len(word) == 6 else word)


def query_terms(question: str) -> list[str]:
    words = TOKEN.findall(normalized(question))
    return list(dict.fromkeys(_concept(w) for w in words
                              if w not in STOPWORDS and (len(w) >= 3 or w == 'ик')))[:12]


def _variants(term: str) -> tuple[str, ...]:
    return CONCEPTS.get(term, (term,))


def fts_words(question: str) -> list[str]:
    return list(dict.fromkeys(v for term in query_terms(question) for v in _variants(term)
                              if re.fullmatch(r'[а-яa-z0-9]+', v)))


def _expression(term: str) -> str:
    # Terms come from a token allowlist, never from user FTS syntax.
    variants = [f'"{v}"*' for v in _variants(term) if re.fullmatch(r'[а-яa-z0-9]+', v)]
    return '(' + ' OR '.join(variants) + ')'


def _present(term: str, words: set[str]) -> bool:
    return any(any(w == v if len(v) <= 2 else w.startswith(v) for w in words)
               for v in _variants(term))


def message_parts(content: str) -> list[tuple[str, str]]:
    markers = list(MESSAGE.finditer(content))
    if not markers:
        return [('', content.strip())] if content.strip() else []
    return [(m.group(1), content[m.end():markers[i+1].start() if i+1 < len(markers) else len(content)].strip())
            for i, m in enumerate(markers)]


def _body(content: str) -> str:
    return '\n'.join(body for _, body in message_parts(content))


def _keys(item: dict) -> set[str]:
    return {ident for ident, _ in message_parts(item['content']) if ident}


def _terms_in(text: str, terms: list[str]) -> set[str]:
    words = set(TOKEN.findall(normalized(text)))
    return {t for t in terms if _present(t, words)}


def _rank(row: dict, terms: list[str], weights: dict[str, float]) -> dict | None:
    parts = message_parts(row['content'])
    bodies = [body for _, body in parts]
    present = _terms_in(' '.join(bodies), terms)
    if {'долго', 'сушка'} <= set(terms) and not {'долго', 'сушка'} <= present:
        return None
    coverage = sum(weights[t] for t in present) / sum(weights.values())
    # A lone generic word must not turn an unrelated question into an answer.
    required = 1 if len(terms) == 1 else 2
    if len(present) < required or coverage < .46:
        return None
    local = max((sum(weights[t] for t in _terms_in(' '.join(bodies[max(0,i-1):i+2]), terms))
                 / sum(weights.values()) for i in range(len(bodies))), default=0)
    body = ' '.join(bodies)
    useful = re.sub(r'\[[^]]*(?:Вложение|содержания)[^]]*\]|https?://\S+', '', body, flags=re.I)
    if len(TOKEN.findall(useful)) < 10:
        return None
    explanatory = bool(re.search(r'потому|если|проверь|попроб|нужно|слишком|необходимо|зависит|совету|причин|лучше', normalized(useful)))
    sentence_focus = max((sum(weights[t] for t in _terms_in(s, terms)) / sum(weights.values())
                          for s in re.split(r'(?<=[.!?])\s+|\n', body)), default=0)
    score = 5 * coverage + 2 * local + 1.5 * sentence_focus + .55 * explanatory
    if {'долго', 'сушка'} <= set(terms):
        symptom = bool(re.search(r'не\s+(?:со[хш]|выс[оы]х)|(?:долг\w*|дольш\w*|медлен\w*)[^.!?]{0,70}(?:сох|суш|высых)', normalized(body)))
        score += .8 * symptom
        if 'кисть' in terms and re.search(r'аэрограф|аэро\b|дюз|распыл', normalized(body)) and 'кисть' not in present:
            score -= 3
    score += .35 * min(1, len(useful) / 700)
    score -= .4 * min(3, len(re.findall(r'https?://', body)))
    # BM25 is a tiebreaker: it must not overwhelm actual concept coverage.
    score += .25 * max(0, -row.get('raw_score', 0)) / (8 + max(0, -row.get('raw_score', 0)))
    return {**row, 'coverage': coverage, 'matched_terms': sorted(present), 'relevance': score}


def search_db(path: str | Path, question: str, limit: int = 5,
              extra_ids: list[int] | None = None) -> list[dict]:
    terms = query_terms(question)
    if not terms or limit <= 0:
        return []
    conn = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    try:
        counts = {t: conn.execute('SELECT COUNT(*) FROM passages_fts WHERE passages_fts MATCH ?',
                                 (_expression(t),)).fetchone()[0] for t in terms}
        total = conn.execute('SELECT COUNT(*) FROM passages').fetchone()[0]
        weights = {t: min(4.5, 1 + math.log((total+1)/(counts[t]+1))) * (.4 if t == 'нанесение' else .65 if t in ('кисть', 'аэрограф') else .55 if t in ('скол', 'царапина', 'лак') else 1) for t in terms}
        candidates = {}
        expressions = [_expression(t) for t in terms]
        queries = [' OR '.join(expressions)]
        # Focused queries prevent common terms from consuming the entire recall pool.
        rare = sorted(terms, key=lambda t: counts[t])[:6]
        queries += [f'{_expression(a)} AND {_expression(b)}' for a, b in combinations(rare, 2)]
        for i, query in enumerate(queries):
            rows = conn.execute('''SELECT p.*, bm25(passages_fts) AS raw_score
                FROM passages_fts JOIN passages p ON p.id=passages_fts.rowid
                WHERE passages_fts MATCH ? ORDER BY raw_score LIMIT ?''',
                                (query, 250 if i == 0 else 40)).fetchall()
            for row in rows:
                candidates.setdefault(row['id'], dict(row))
        if extra_ids:
            for ident in extra_ids[:40]:
                row = conn.execute('SELECT *, 0 AS raw_score FROM passages WHERE id=?', (ident,)).fetchone()
                if row:
                    candidates[ident] = dict(row)
    finally:
        conn.close()
    ranked = [item for row in candidates.values() if (item := _rank(row, terms, weights))]
    ranked.sort(key=lambda r: (-r['relevance'], r['id']))
    selected, seen_threads = [], Counter()
    while ranked and len(selected) < limit:
        def novelty(item):
            keys = _keys(item)
            overlap = max((len(keys & _keys(s))/max(1, min(len(keys), len(_keys(s))))
                           for s in selected), default=0)
            return item['relevance'] - 3 * overlap
        ranked.sort(key=lambda r: (-novelty(r), r['id']))
        item = ranked.pop(0)
        if seen_threads[item['thread_id']] >= 2:
            continue
        keys = _keys(item)
        if keys and any(len(keys & _keys(s))/min(len(keys), len(_keys(s))) > .55
                        for s in selected if _keys(s)):
            continue
        # Duplicate excerpts also occur in different source threads.
        if any(normalized(_body(item['content'])) == normalized(_body(s['content'])) for s in selected):
            continue
        selected.append(item)
        seen_threads[item['thread_id']] += 1
    return selected


def context_for(question: str, hits: list[dict], max_chars: int = 7000) -> str:
    """Extract relevant messages + their neighbours, without source labels/IDs.

    Whole short messages are retained. For a long message the relevant sentences,
    with their neighbours, win over an arbitrary first-1800-characters slice.
    """
    if max_chars <= 0:
        return ''
    terms = query_terms(question)
    seen_ids, seen_text = set(), set()
    groups = []
    for item in hits:
        parts = message_parts(item['content'])
        anchors = [i for i, (_, body) in enumerate(parts) if _terms_in(body, terms)]
        wanted = {j for i in anchors for j in range(max(0, i-1), min(len(parts), i+3))}
        if not wanted:
            continue
        notes = []
        for i in sorted(wanted):
            ident, body = parts[i]
            key = normalized(body)
            if (ident and ident in seen_ids) or key in seen_text:
                continue
            if not re.search(r'[а-яa-z]', body, re.I):
                continue
            if len(body) > 1800:
                sentences = re.split(r'(?<=[.!?])\s+|\n+', body)
                relevant = [j for j, s in enumerate(sentences) if _terms_in(s, terms)]
                indices = {k for j in relevant for k in range(max(0,j-1), min(len(sentences),j+2))}
                body = ' '.join(sentences[j] for j in sorted(indices)) or body
            # Never cut a sentence to squeeze it into the budget. If a single
            # sentence cannot fit, leave it out rather than invert its meaning.
            available = max_chars - sum(len(g)+2 for g in groups) - sum(len(n)+1 for n in notes)
            if len(body) > available:
                sentences = re.split(r'(?<=[.!?])\s+|\n+', body)
                chosen = []
                for sentence in sentences:
                    if len(sentence) + sum(len(s)+1 for s in chosen) <= available:
                        chosen.append(sentence)
                body = ' '.join(chosen)
            if not body:
                continue
            notes.append(body)
            if ident:
                seen_ids.add(ident)
            seen_text.add(key)
        if notes:
            groups.append('\n'.join(notes))
    return '\n\n'.join(groups)[:max_chars]
