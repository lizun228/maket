"""Offline Russian-friendly FTS5 lookup of sanitized Telegram discussion excerpts."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

STOPWORDS = set('и а но или в во на по за с со до от для при из к ко о об про это что как чем где когда если есть нет да вы мы они она он его её их нас вас мне тебе тут там уже еще ещё только так вообще очень надо нужно можно будет была были был быть почему после перед вопрос здравствуйте подскажите пожалуйста добрый день вечер ребята всем привет делать сделать делаю сделаю делаете сделали какой какая какие который которая которые ли же то тут'.split())
# Lexical variants improve retrieval without extra dependencies or remote indexing.
ALIASES = {
    'царап': ('царап', 'рисок', 'риски', 'поврежд'),
    'скол': ('скол', 'камн', 'поврежд'),
    'подбор': ('подбор', 'колеров', 'цвет'),
    'цвет': ('цвет', 'оттен', 'колер', 'подбор'),
    'полир': ('полир', 'паст', 'круг'),
    'лак': ('лак', 'лаков'),
    'суш': ('суш', 'ламп', 'ик'),
    'просад': ('просад', 'усад', 'провал'),
    'грунт': ('грунт', 'подлож'),
    'шлиф': ('шлиф', 'наждак', 'абразив'),
    'инфрак': ('инфрак', 'ламп', 'ик'),
    'срез': ('срез', 'каттер', 'резак'),
}


def query_terms(question: str) -> list[str]:
    raw = re.findall(r'[а-яёa-z0-9]+', question.lower().replace('ё', 'е'))
    raw = [x for x in raw if len(x) >= 4 and x not in STOPWORDS]
    output = []
    for word in raw[:12]:
        stem = word[:5] if len(word) >= 7 else (word[:4] if len(word) == 6 else word)
        output.append(stem)
    if len(output) < 2:
        output.extend(w[:5] for w in raw if len(w) > 4)
    return list(dict.fromkeys(output))[:10]


def fts_words(question: str) -> list[str]:
    terms = query_terms(question)
    expanded = list(terms)
    for term in terms:
        for key, variants in ALIASES.items():
            if key.startswith(term) or term.startswith(key[:4]):
                expanded.extend(variants)
    unique = []
    for word in expanded:
        if re.fullmatch(r'[а-яa-z0-9]+', word) and word not in unique:
            unique.append(word)
    return unique[:24]


def search_db(path: str | Path, question: str, limit: int = 5) -> list[dict]:
    terms = fts_words(question)
    if not terms:
        return []
    conn = sqlite3.connect(f'file:{Path(path).resolve().as_posix()}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    try:
        # Restrict results to relevant common and exact variants; OR is for recall.
        matches = ' OR '.join(f'{term}*' for term in terms)
        results = conn.execute('''
            SELECT p.id, p.thread_id, p.start_message_id, p.end_message_id,
                   p.content, p.contains_media, bm25(passages_fts) AS raw_score
            FROM passages_fts JOIN passages p ON p.id = passages_fts.rowid
            WHERE passages_fts MATCH ? ORDER BY raw_score LIMIT 100
        ''', (matches,)).fetchall()
    finally:
        conn.close()
    scored = []
    core = query_terms(question)
    for row in results:
        content = row['content'].lower().replace('ё', 'е')
        hits = sum(bool(re.search(r'(?<![а-яa-z])' + re.escape(t), content)) for t in core)
        if not hits:
            # Alias-only matches can be useful but should score much lower.
            hits = .25
        coverage = hits / max(1, len(core))
        # Deprioritize chunks with one match in a longer question.
        if len(core) >= 3 and hits < 2:
            continue
        raw = dict(row)
        raw['coverage'] = coverage
        raw['relevance'] = (coverage * 5) + min(5, max(0, -row['raw_score'])) * .35
        scored.append(raw)
    scored.sort(key=lambda x: x['relevance'], reverse=True)
    selected, seen_threads = [], {}
    for item in scored:
        if seen_threads.get(item['thread_id'], 0) >= 2:
            continue
        selected.append(item)
        seen_threads[item['thread_id']] = seen_threads.get(item['thread_id'], 0) + 1
        if len(selected) >= limit:
            break
    return selected


def context_for(question: str, hits: list[dict]) -> str:
    result = []
    for n, item in enumerate(hits, 1):
        content = item['content'][:1800]
        result.append(f'[Источник {n}: {item["thread_id"]}, сообщения {item["start_message_id"]}-{item["end_message_id"]}]\n{content}')
    return '\n\n---\n\n'.join(result)
