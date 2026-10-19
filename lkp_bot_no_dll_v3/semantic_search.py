"""Optional local semantic recall. All cache files are derived, never the archive.

No model is downloaded automatically. No network call leaves loopback. FTS remains
usable without an embedding model or when its version/cache is unavailable.
"""
from __future__ import annotations
from contextlib import closing
from array import array
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from local_models import embed
from search_engine import _body


def fingerprint(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def unit_vector(values) -> array:
    if not values or not all(isinstance(x, (float, int)) and math.isfinite(x) for x in values):
        raise ValueError('Некорректный вектор')
    norm = math.sqrt(sum(x*x for x in values))
    if norm == 0:
        raise ValueError('Нулевой вектор')
    return array('f', (x/norm for x in values))


def model_digest(base: str, model: str) -> str:
    import httpx
    from local_models import ollama_url
    with httpx.Client(timeout=10, trust_env=False, follow_redirects=False) as client:
        response = client.get(ollama_url(base, '/api/tags'))
        response.raise_for_status()
        names = {model, model + ':latest'}
        for item in response.json().get('models', []):
            if item.get('name') in names or item.get('model') in names:
                if item.get('digest'):
                    return item['digest']
    raise ValueError('Embedding-модель не установлена в локальной Ollama')


def build_vectors(source: Path, target: Path, base: str, model: str,
                  replace: bool = False, batch_size: int = 16) -> int:
    import os
    import shutil
    from datetime import datetime, timezone
    source, target = source.resolve(), target.resolve()
    if source == target:
        raise ValueError('Нельзя записывать векторный индекс поверх архива')
    if target.exists() and not replace:
        raise FileExistsError('Индекс уже существует; для обновления укажи --replace')
    if target.exists():
        with closing(sqlite3.connect(target.as_uri()+'?mode=ro', uri=True)) as existing:
            if not existing.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='vectors'").fetchone():
                raise ValueError('Файл назначения не является векторным индексом; заменять его нельзя')
    digest = model_digest(base, model)
    metadata = {'source_sha256': fingerprint(source), 'model': model, 'model_digest': digest,
                'format': 1}
    partial = target.with_name(target.name + '.partial')
    if partial.exists():
        with closing(sqlite3.connect(partial.resolve().as_uri()+'?mode=ro', uri=True)) as check:
            tables = {r[0] for r in check.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables != {'metadata', 'vectors'}:
                raise ValueError('Путь незавершённого индекса занят другим файлом')
    with closing(sqlite3.connect(partial)) as out, out:
        out.executescript('CREATE TABLE IF NOT EXISTS metadata(data TEXT NOT NULL); CREATE TABLE IF NOT EXISTS vectors(id INTEGER PRIMARY KEY, vector BLOB NOT NULL);')
        old = out.execute('SELECT data FROM metadata').fetchone()
        if old and json.loads(old[0]) != metadata:
            raise ValueError('Незавершённый индекс относится к другой базе/модели. Сохрани его под другим именем перед новой индексацией.')
        if not old:
            out.execute('INSERT INTO metadata VALUES(?)', (json.dumps(metadata),))
            out.commit()
        completed = {row[0] for row in out.execute('SELECT id FROM vectors')}
        with closing(sqlite3.connect(source.as_uri()+'?mode=ro', uri=True)) as src:
            rows = [(i, _body(body)) for i, body in src.execute('SELECT id,content FROM passages ORDER BY id') if i not in completed]
            expected = src.execute('SELECT count(*) FROM passages').fetchone()[0]
        dims = None
        first = out.execute('SELECT length(vector) FROM vectors LIMIT 1').fetchone()
        if first:
            dims = first[0]//array('f').itemsize
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start+batch_size]
            vectors = embed(base, model, [body for _, body in batch])
            for (ident, _), vector in zip(batch, vectors):
                values = unit_vector(vector)
                dims = dims or len(values)
                if len(values) != dims:
                    raise ValueError('Изменилась размерность модели')
                out.execute('INSERT INTO vectors VALUES(?,?)', (ident, values.tobytes()))
            out.commit()
            print(f'Индексировано: {len(completed)+min(start+batch_size,len(rows))}/{expected}', flush=True)
        actual = out.execute('SELECT count(*) FROM vectors').fetchone()[0]
        if actual != expected or fingerprint(source) != metadata['source_sha256']:
            raise ValueError('Архив изменился во время индексации')
        if model_digest(base, model) != digest:
            raise ValueError('Модель обновилась во время индексации')
    if target.exists():
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        shutil.copy2(target, target.with_name(target.name + '.' + stamp + '.bak'))
    os.replace(partial, target)
    return actual


@lru_cache(maxsize=2)
def _load(source_name: str, source_stat: tuple, cache_name: str, cache_stat: tuple, model: str):
    source, cache = Path(source_name), Path(cache_name)
    with closing(sqlite3.connect(cache.as_uri()+'?mode=ro', uri=True)) as con:
        metadata = json.loads(con.execute('SELECT data FROM metadata').fetchone()[0])
        if metadata.get('source_sha256') != fingerprint(source) or metadata.get('model') != model:
            raise ValueError('Векторный индекс устарел; перестрой его')
        vectors = []
        for ident, blob in con.execute('SELECT id,vector FROM vectors'):
            values = array('f')
            values.frombytes(blob)
            vectors.append((ident, values))
    return metadata, vectors


def nearest(source: Path, cache: Path, question: str, base: str, model: str, limit: int = 30) -> list[tuple[int, float]]:
    source, cache = source.resolve(), cache.resolve()
    def signature(path):
        stat = path.stat()
        return (stat.st_size, stat.st_mtime_ns)
    metadata, vectors = _load(str(source), signature(source), str(cache), signature(cache), model)
    if model_digest(base, model) != metadata['model_digest']:
        raise ValueError('Embedding-модель изменилась; перестрой индекс')
    query = unit_vector(embed(base, model, [question])[0])
    if any(len(vector) != len(query) for _, vector in vectors):
        raise ValueError('Не совпадает размерность векторов')
    scores = [(ident, sum(a*b for a,b in zip(query, vector))) for ident, vector in vectors]
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores[:limit]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Необязательный локальный смысловой индекс через Ollama')
    parser.add_argument('--source', type=Path, default=Path(__file__).with_name('knowledge.sqlite'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('embeddings.sqlite'))
    parser.add_argument('--base-url', default='http://127.0.0.1:11434')
    parser.add_argument('--model', default='bge-m3')
    parser.add_argument('--replace', action='store_true')
    args = parser.parse_args()
    print('Готово:', build_vectors(args.source, args.output, args.base_url, args.model, args.replace))
