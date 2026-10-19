"""Non-destructive local indexing of JSONL threads or a full Telegram JSON export."""
from __future__ import annotations
from contextlib import closing
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile


SCHEMA = '''
CREATE TABLE passages(
 id INTEGER PRIMARY KEY, thread_id TEXT NOT NULL,
 start_message_id INTEGER NOT NULL, end_message_id INTEGER NOT NULL,
 content TEXT NOT NULL, contains_media INTEGER NOT NULL, search_text TEXT NOT NULL
);
CREATE VIRTUAL TABLE passages_fts USING fts5(
 search_text, content='passages', content_rowid='id', tokenize='unicode61 remove_diacritics 2'
);
CREATE INDEX idx_thread ON passages(thread_id);
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
'''


def clean_message(text) -> str:
    if isinstance(text, list):
        text = ''.join(item if isinstance(item, str) else str(item.get('text', ''))
                       for item in text if isinstance(item, (str, dict)))
    text = re.sub(r'\[Вложение:[^\]]+\]', '[Вложение без доступного для анализа содержания]', str(text or ''))
    # Preserve full long answers and paragraph boundaries, including qualifications.
    return re.sub(r'[ \t]+', ' ', text).strip()


def windows(thread: dict) -> list[list[dict]]:
    msgs = []
    for msg in thread.get('messages', []):
        body = clean_message(msg.get('text'))
        if not body or not re.search(r'\w', body):
            continue
        msgs.append({'id': int(msg.get('message_id', msg.get('id'))), 'text': body,
                     'has_media': bool(msg.get('has_media')),
                     'reply_to': msg.get('reply_to_message_id')})
    return [msgs[max(0, i-2):i+8] for i in range(0, len(msgs), 6)]


def _telegram_threads(data: dict):
    messages = []
    for msg in data.get('messages', []):
        if msg.get('type', 'message') != 'message':
            continue
        body = clean_message(msg.get('text'))
        media = any(msg.get(k) for k in ('photo', 'file', 'media_type'))
        if media:
            body += '\n[Вложение без доступного для анализа содержания]'
        if not body.strip():
            continue
        messages.append({'message_id': int(msg['id']), 'text': body.strip(),
                         'has_media': media, 'reply_to_message_id': msg.get('reply_to_message_id'),
                         'date': msg.get('date_unixtime') or msg.get('date')})
    messages.sort(key=lambda m: m['message_id'])
    by_id = {m['message_id']: m for m in messages}
    session, previous_time = [], None
    for msg in messages:
        value = msg['date']
        try:
            stamp = float(value) if str(value).isdigit() else datetime.fromisoformat(value).timestamp()
        except (ValueError, TypeError):
            stamp = previous_time
        if session and stamp is not None and previous_time is not None and stamp - previous_time > 45*60:
            yield _with_parents(session, by_id)
            session = []
        session.append(msg)
        previous_time = stamp
    if session:
        yield _with_parents(session, by_id)


def _with_parents(session: list[dict], by_id: dict) -> dict:
    # Explicit reply chains can cross a long pause. Retain their parent messages.
    # Add them in windows(), not at the beginning of an arbitrarily huge session.
    return {'thread_id': f'telegram:{session[0]["message_id"]}', 'messages': session,
            '_parents': by_id}


def read_threads(source: Path, source_format: str):
    if source_format == 'auto':
        source_format = 'jsonl' if source.suffix.lower() in ('.jsonl', '.ndjson') else 'telegram'
    with source.open('r', encoding='utf-8-sig') as handle:
        if source_format == 'jsonl':
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        else:
            data = json.load(handle)
            if not isinstance(data, dict) or not isinstance(data.get('messages'), list):
                raise ValueError('Ожидается JSON-экспорт одного Telegram-чата с массивом messages')
            yield from _telegram_threads(data)


def build(source: Path, target: Path, replace: bool = False, source_format: str = 'auto') -> tuple[int, int]:
    source, target = source.resolve(), target.resolve()
    if source == target:
        raise ValueError('Исходник и индекс должны быть разными файлами')
    if target.exists() and not replace:
        raise FileExistsError('Индекс уже существует. Выбери новый --output или укажи --replace (с резервной копией).')
    if target.exists():
        with closing(sqlite3.connect(target.as_uri()+'?mode=ro', uri=True)) as existing:
            if not existing.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='passages'").fetchone():
                raise ValueError('Файл назначения не является базой знаний. Данные доступа и другие базы заменять нельзя.')
    with source.open('rb') as f:
        source_hash = hashlib.file_digest(f, 'sha256').hexdigest()
    fd, temp_name = tempfile.mkstemp(prefix=target.name+'.', suffix='.partial', dir=target.parent)
    os.close(fd)
    temp = Path(temp_name)
    count = n_threads = 0
    message_ids = set()
    try:
        with closing(sqlite3.connect(temp)) as conn, conn:
            conn.executescript(SCHEMA)
            for thread in read_threads(source, source_format):
                n_threads += 1
                for chunk in windows(thread):
                    parents = thread.get('_parents', {})
                    augmented = {m['id']: m for m in chunk}
                    for msg in list(chunk):
                        parent = msg.get('reply_to')
                        visited = set()
                        for _ in range(4):
                            if parent in visited or parent not in parents:
                                break
                            visited.add(parent)
                            original = parents[parent]
                            augmented[parent] = {'id': parent, 'text': original['text'],
                                                 'has_media': original['has_media']}
                            parent = original.get('reply_to_message_id')
                    chunk = sorted(augmented.values(), key=lambda m: m['id'])
                    content = '\n'.join(f'№{m["id"]}: {m["text"]}' for m in chunk)
                    search = ' '.join(m['text'] for m in chunk).casefold().replace('ё', 'е')
                    cursor = conn.execute('''INSERT INTO passages(thread_id,start_message_id,end_message_id,content,contains_media,search_text)
                        VALUES(?,?,?,?,?,?)''', (str(thread['thread_id']), chunk[0]['id'], chunk[-1]['id'], content,
                                               int(any(m['has_media'] for m in chunk)), search))
                    conn.execute('INSERT INTO passages_fts(rowid,search_text) VALUES(?,?)', (cursor.lastrowid, search))
                    count += 1
                    message_ids.update(m['id'] for m in chunk)
            if count == 0:
                raise ValueError('В исходнике нет сообщений: существующая база не изменена')
            with source.open('rb') as f:
                if hashlib.file_digest(f, 'sha256').hexdigest() != source_hash:
                    raise ValueError('Исходный файл изменился во время индексации')
            meta = {'source_sha256': source_hash, 'unique_messages': str(len(message_ids)), 'format_version': '2'}
            conn.executemany('INSERT INTO metadata VALUES(?,?)', meta.items())
            conn.execute("INSERT INTO passages_fts(passages_fts,rank) VALUES('integrity-check',1)")
            if conn.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise RuntimeError('Проверка нового индекса не пройдена')
        if target.exists():
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            backup = target.with_name(target.name + '.' + stamp + '.bak')
            with closing(sqlite3.connect(target.as_uri()+'?mode=ro',uri=True)) as old, closing(sqlite3.connect(backup)) as saved, saved:
                old.backup(saved)
        os.replace(temp, target)
        return n_threads, count
    finally:
        temp.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('lkp_candidate_discussions.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('knowledge.sqlite'))
    parser.add_argument('--format', choices=('auto', 'jsonl', 'telegram'), default='auto')
    parser.add_argument('--replace', action='store_true', help='Сначала проверяет новую базу, затем делает backup старой и заменяет её')
    args = parser.parse_args()
    threads, passages = build(args.source, args.output, args.replace, args.format)
    print(f'Готово: {threads} обсуждений/сеансов, {passages} отрывков -> {args.output}')
