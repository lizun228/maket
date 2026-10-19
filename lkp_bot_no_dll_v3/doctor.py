"""Safe local diagnostics: prints presence/status, never secret values or messages."""
from __future__ import annotations
from contextlib import closing
import argparse
from pathlib import Path
import os
import sqlite3
import sys
from dotenv import load_dotenv
import httpx
from local_models import ollama_url
from grounded_answers import ANSWER_VERSION

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-ollama',action='store_true')
    args=parser.parse_args()
    load_dotenv(ROOT/'.env')
    print('Версия проверки ответов:',ANSWER_VERSION)
    failures=[]
    def check(label,good):
        print(('OK: ' if good else 'Нужно исправить: ')+label)
        if not good: failures.append(label)
    check('Python 3.11 или новее',sys.version_info>=(3,11))
    try:
        import telegram
        check('Библиотеки Telegram/httpx/dotenv доступны',True)
    except ImportError:
        check('Библиотеки Telegram/httpx/dotenv доступны',False)
    token=bool(os.environ.get('TELEGRAM_BOT_TOKEN','').strip())
    check('Telegram-токен задан (значение скрыто)',token)
    admins=os.environ.get('ADMIN_TELEGRAM_IDS','').strip()
    check('Числовой ID администратора задан',bool(admins) and all(x.strip().isdigit() for x in admins.split(',')))
    check('Локальный режим Ollama выбран',os.environ.get('AI_PROVIDER','ollama').lower()=='ollama')
    try:
        path=ROOT/'knowledge.sqlite'
        with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as con:
            good=con.execute('PRAGMA quick_check').fetchone()[0]=='ok'
            count=con.execute('SELECT count(*) FROM passages').fetchone()[0]
            con.execute("SELECT count(*) FROM passages_fts WHERE passages_fts MATCH 'скол*'").fetchone()
        check('База знаний и FTS5 читаются',good and count>0)
        print('Фрагментов в базе:',count)
    except (OSError,sqlite3.Error):
        check('База знаний и FTS5 читаются',False)
    if not args.skip_ollama:
        try:
            base=os.environ.get('OLLAMA_BASE_URL','http://127.0.0.1:11434')
            model=os.environ.get('AI_MODEL','qwen3:4b-instruct')
            with httpx.Client(timeout=10,trust_env=False,follow_redirects=False) as client:
                response=client.get(ollama_url(base,'/api/tags'))
                response.raise_for_status()
                names={m['name'] for m in response.json().get('models',[])}
            check('Локальная Ollama доступна',True)
            check('Модель из AI_MODEL установлена',model in names or model+':latest' in names)
        except Exception as exc:
            check('Локальная Ollama доступна',False)
            print('Тип ошибки:',type(exc).__name__)
    print('Диагностика не отправляет сообщения в Telegram и не запускает генерацию.')
    return 1 if failures else 0


if __name__=='__main__':
    raise SystemExit(main())
