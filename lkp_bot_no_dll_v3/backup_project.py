"""Create a local recoverable backup, including secrets and consistent SQLite copies.

Keep the backup private. Nothing is uploaded. Run with the bot stopped before an update.
"""
from __future__ import annotations
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

ROOT=Path(__file__).resolve().parent


def backup(root: Path=ROOT) -> Path:
    root=root.resolve()
    parent=root.parent/(root.name+'_backups')
    parent.mkdir(exist_ok=True)
    target=parent/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    databases=list(root.glob('*.sqlite'))
    ignored={'__pycache__','.git','.venv','.venv312','backups'}
    ignored.update(p.name for p in databases)
    def ignore(folder,names):
        return [name for name in names if name in ignored or name.endswith(('.sqlite-wal','.sqlite-shm'))]
    shutil.copytree(root,target,ignore=ignore)
    for source in databases:
        with closing(sqlite3.connect(source.as_uri()+'?mode=ro',uri=True)) as src, closing(sqlite3.connect(target/source.name)) as dest:
            src.backup(dest)
            if dest.execute('PRAGMA quick_check').fetchone()[0]!='ok':
                raise RuntimeError('Резервная копия базы не прошла проверку')
    hashes={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in target.rglob('*') if p.is_file()}
    (target/'backup_manifest.json').write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
    if not all(hashlib.sha256((target/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items()):
        raise RuntimeError('Проверка файлов резервной копии не пройдена')
    return target


if __name__=='__main__':
    print('Резервная копия проверена:',backup())
    print('Она может содержать .env и личные данные. Храни её локально и не отправляй посторонним.')
