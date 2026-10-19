"""Build a private, local SQLite database from candidate discussion threads."""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path


def clean_message(text: str) -> str:
    text = re.sub(r'\[Вложение:[^\]]+\]', '[Вложение без доступного для анализа содержания]', str(text or ''))
    return re.sub(r'\s+', ' ', text).strip()[:500]


def windows(thread: dict) -> list[list[dict]]:
    msgs = []
    for msg in thread.get('messages', []):
        body = clean_message(msg.get('text'))
        if not body or body in ('👍', '👌', '😂', '😊', '🙏'):
            continue
        msgs.append({'id': int(msg['message_id']), 'text': body,
                     'has_media': bool(msg.get('has_media'))})
    if not msgs:
        return []
    output = []
    # Overlapping local turns preserve follow-up/answer context without indexing huge conversations.
    for i in range(0, len(msgs), 6):
        window = msgs[max(0, i-2): i+8]
        if sum(len(x['text']) for x in window) < 100:
            continue
        output.append(window)
    return output


def build(source: Path, target: Path) -> tuple[int,int]:
    if target.exists():
        target.unlink()
    conn = sqlite3.connect(target)
    conn.executescript('''
        PRAGMA journal_mode=DELETE;
        CREATE TABLE passages(
            id INTEGER PRIMARY KEY, thread_id TEXT NOT NULL,
            start_message_id INTEGER NOT NULL, end_message_id INTEGER NOT NULL,
            content TEXT NOT NULL, contains_media INTEGER NOT NULL, search_text TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE passages_fts USING fts5(
            search_text, content='passages', content_rowid='id', tokenize='unicode61 remove_diacritics 2'
        );
        CREATE INDEX idx_thread ON passages(thread_id);
    ''')
    count, n_threads = 0, 0
    with source.open('r', encoding='utf-8') as handle:
        for line in handle:
            thread = json.loads(line)
            n_threads += 1
            for chunk in windows(thread):
                content = '\n'.join(f'№{m["id"]}: {m["text"]}' for m in chunk)
                # Raw text may include quoted replies but sender identifiers are not stored.
                index_text = ' '.join(m['text'] for m in chunk).replace('ё', 'е').replace('Ё','Е')
                is_media = any(m['has_media'] for m in chunk)
                cur = conn.execute('''
                   INSERT INTO passages(thread_id,start_message_id,end_message_id,
                       content,contains_media,search_text) VALUES(?,?,?,?,?,?)
                ''', (thread['thread_id'], chunk[0]['id'], chunk[-1]['id'],
                      content, int(is_media), index_text))
                conn.execute('INSERT INTO passages_fts(rowid,search_text) VALUES(?,?)',
                             (cur.lastrowid,index_text))
                count += 1
    conn.commit()
    conn.close()
    return n_threads,count


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='lkp_candidate_discussions.jsonl')
    parser.add_argument('--output', default='knowledge.sqlite')
    a = parser.parse_args()
    thread_count, passage_count = build(Path(a.source), Path(a.output))
    print(f'Готово: {thread_count} обсуждений, {passage_count} отрывков -> {a.output}')
