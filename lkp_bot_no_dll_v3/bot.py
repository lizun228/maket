"""Paid-access Paint Restoration Q&A assistant. Run locally using Telegram polling."""
from __future__ import annotations

import asyncio
from contextlib import suppress, contextmanager
from datetime import datetime, timedelta, timezone
import logging
import os
from pathlib import Path
import sqlite3

from dotenv import load_dotenv
import httpx
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from search_engine import search_db, context_for
from conversation import load_turns, search_question, remember
from local_models import chat
from qa_pipeline import produce_answer
from response_quality import SYSTEM, clean_answer, NO_EVIDENCE, split_telegram, prompt_budget

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / '.env')
TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN','').strip()
PROVIDER = os.environ.get('AI_PROVIDER', 'ollama').lower().strip()
if PROVIDER not in ('proxyapi','ollama','offline'):
    raise SystemExit('AI_PROVIDER должен быть proxyapi, ollama или offline')
KEY = os.environ.get('AI_API_KEY','').strip()
MODEL = os.environ.get('AI_MODEL', 'deepseek/deepseek-chat' if PROVIDER == 'proxyapi' else 'qwen3:4b-instruct').strip()
BASE_URL = {
    'proxyapi': 'https://api.proxyapi.ru/v1/chat/completions',
    'ollama': os.environ.get('OLLAMA_BASE_URL', 'http://127.0.0.1:11434'),
    'offline': '',
}[PROVIDER]

KNOWLEDGE = BASE / 'knowledge.sqlite'
ACCESS_DB = BASE / 'access.sqlite'
ADMIN_ID_INPUT = os.environ.get('ADMIN_TELEGRAM_IDS', '').strip()
ADMINS = {int(x.strip()) for x in ADMIN_ID_INPUT.split(',') if x.strip().isdigit()}
MAX_DAILY = int(os.environ.get('DAILY_QUESTION_LIMIT','25'))

LOG = logging.getLogger('lkp_bot')


MODEL_GATE = asyncio.Semaphore(1)  # one generation at a time on an 8 GB GPU
ALLOW_EXTERNAL = os.environ.get('ALLOW_EXTERNAL_AI', 'false').lower() == 'true'
NUM_CTX = int(os.environ.get('OLLAMA_NUM_CTX', '8192'))
NUM_PREDICT = int(os.environ.get('OLLAMA_NUM_PREDICT', '650'))
SEMANTIC = os.environ.get('SEMANTIC_SEARCH', 'false').lower() == 'true'


async def generate_response(system: str, prompt: str) -> str:
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': prompt}]
    if PROVIDER == 'ollama':
        return await chat(BASE_URL, MODEL, messages, num_ctx=NUM_CTX, num_predict=NUM_PREDICT)
    if PROVIDER != 'proxyapi':
        raise ValueError('No AI available in offline mode')
    if not ALLOW_EXTERNAL:
        raise ValueError('Передача переписки внешнему ИИ выключена: ALLOW_EXTERNAL_AI=false')
    async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=10)) as client:
        response = await client.post(BASE_URL, json={
            'model': MODEL, 'messages': messages, 'max_tokens': NUM_PREDICT, 'temperature': .2,
        }, headers={'Authorization': f'Bearer {KEY}'})
        response.raise_for_status()
        choices = response.json().get('choices', [])
    text = choices[0].get('message', {}).get('content', '') if choices else ''
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError('Модель вернула пустой ответ')
    return text.strip()


async def retrieve(question: str) -> list[dict]:
    if not SEMANTIC:
        return await asyncio.to_thread(search_db, KNOWLEDGE, question, 6)
    try:
        from semantic_search import nearest
        neighbours = await asyncio.to_thread(
            nearest, KNOWLEDGE, BASE / 'embeddings.sqlite', question,
            os.environ.get('OLLAMA_BASE_URL', 'http://127.0.0.1:11434'),
            os.environ.get('EMBEDDING_MODEL', 'bge-m3'))
    except Exception as exc:
        # Do not log request bodies, archive fragments, URLs or credentials.
        LOG.warning('Смысловой индекс недоступен (%s); используется текстовый поиск', type(exc).__name__)
        return await asyncio.to_thread(search_db, KNOWLEDGE, question, 6)
    candidates = await asyncio.to_thread(search_db, KNOWLEDGE, question, 15,
                                        [ident for ident, _ in neighbours])
    scores = dict(neighbours)
    candidates.sort(key=lambda row: -(row['relevance'] + .6 * max(0, scores.get(row['id'], 0))))
    return candidates[:6]


async def keep_typing(context, chat_id):
    while True:
        with suppress(Exception):
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        await asyncio.sleep(4)


@contextmanager
def db():
    con = sqlite3.connect(ACCESS_DB, timeout=10)
    try:
        con.execute('PRAGMA busy_timeout=10000')
        with con:
            yield con
    finally:
        con.close()


def init_db() -> None:
    with db() as c:
        c.executescript('''
          CREATE TABLE IF NOT EXISTS access (
            user_id INTEGER PRIMARY KEY, granted_at TEXT NOT NULL,
            expires_at TEXT NOT NULL, granted_by INTEGER NOT NULL
          );
          CREATE TABLE IF NOT EXISTS usage (
            user_id INTEGER NOT NULL, day TEXT NOT NULL, count INTEGER NOT NULL,
            PRIMARY KEY(user_id,day)
          );
        ''')


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def allowed(user_id: int) -> tuple[bool, str]:
    if user_id in ADMINS:
        return True, 'Администратор'
    with db() as c:
        row = c.execute('SELECT expires_at FROM access WHERE user_id=?',(user_id,)).fetchone()
    if not row:
        return False, 'Доступ пока не открыт.'
    expiry = datetime.fromisoformat(row[0])
    if expiry <= now_utc():
        return False, 'Срок доступа закончился.'
    return True, f'Доступ до {expiry.strftime("%d.%m.%Y")} (UTC)'


def grant(user_id: int, admin_id: int, days: int) -> str:
    expires = now_utc() + timedelta(days=days)
    with db() as c:
        c.execute('''INSERT INTO access(user_id,granted_at,expires_at,granted_by)
                     VALUES(?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET
                     granted_at=excluded.granted_at,expires_at=excluded.expires_at,
                     granted_by=excluded.granted_by''',
                  (user_id,now_utc().isoformat(),expires.isoformat(),admin_id))
    return expires.strftime('%d.%m.%Y')


def revoke(user_id: int) -> bool:
    with db() as c:
        cur = c.execute('DELETE FROM access WHERE user_id=?', (user_id,))
        return cur.rowcount > 0


def consume(user_id: int) -> bool:
    if user_id in ADMINS:
        return True
    date = now_utc().date().isoformat()
    with db() as c:
        c.execute('BEGIN IMMEDIATE')
        row = c.execute('SELECT count FROM usage WHERE user_id=? AND day=?',
                        (user_id,date)).fetchone()
        if row and row[0] >= MAX_DAILY:
            return False
        c.execute('''INSERT INTO usage(user_id,day,count) VALUES(?,?,1)
                    ON CONFLICT(user_id,day) DO UPDATE SET count=count+1''',
                  (user_id,date))
    return True


def refund(user_id: int, day: str | None = None) -> None:
    """An unavailable model must not consume a student's paid daily allowance."""
    if user_id in ADMINS:
        return
    with db() as c:
        c.execute('UPDATE usage SET count=max(0,count-1) WHERE user_id=? AND day=?',
                  (user_id, day or now_utc().date().isoformat()))


async def new_topic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lock = context.user_data.setdefault('_lock', asyncio.Lock())
    async with lock:
        context.user_data.pop('turns', None)
        context.user_data.pop('turns_at', None)
    await update.message.reply_text('Начнём новую тему. Напиши свой вопрос.')


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    yes, info = allowed(uid)
    if yes:
        await update.message.reply_text('Привет! Я помощник по реставрации ЛКП.\n\nПросто задай вопрос текстом, например: «Почему после сушки появляется просадка?»\n\n/status — проверить доступ\n/id — узнать свой Telegram ID')
    else:
        await update.message.reply_text(f'Привет! Я учебный ИИ-помощник по реставрации ЛКП.\n\nТвой Telegram ID: {uid}\n{info}\n\nПосле подтверждения оплаты передай этот ID организатору курса. Он откроет доступ.')


async def my_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f'Твой Telegram ID: {update.effective_user.id}')


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    yes, info = allowed(update.effective_user.id)
    await update.message.reply_text(('✅ ' if yes else '🔒 ') + info)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Напиши свой вопрос по локальной реставрации ЛКП.\n/id — твой Telegram ID\n/status — статус доступа\n/start — начать\n/new — новая тема\n\nПока бот понимает только текст. Для администратора: /grant ID 180, /revoke ID, /stats.')


async def admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if uid not in ADMINS:
        await update.message.reply_text('Команда доступна только администратору.')
        return
    command = update.message.text.split()[0].split('@')[0]
    args = context.args
    if command == '/stats':
        with db() as c:
            active = c.execute('SELECT count(*) FROM access WHERE expires_at > ?', (now_utc().isoformat(),)).fetchone()[0]
            today = c.execute('SELECT coalesce(sum(count),0) FROM usage WHERE day=?',(now_utc().date().isoformat(),)).fetchone()[0]
        await update.message.reply_text(f'Активных учеников: {active}\nЗапросов сегодня: {today}\nРежим: {PROVIDER}\nМодель: {MODEL if PROVIDER != "offline" else "поиск без ИИ"}')
        return
    if not args or not args[0].isdigit() or not (5 <= len(args[0]) <= 15):
        await update.message.reply_text('Формат: /grant 123456789 180\nИли: /revoke 123456789')
        return
    pupil_id = int(args[0])
    if command == '/grant':
        if len(args) > 2 or (len(args) == 2 and not args[1].isdigit()):
            await update.message.reply_text('Формат: /grant ID 180')
            return
        days = int(args[1]) if len(args) == 2 else 180
        if not 1 <= days <= 730:
            await update.message.reply_text('Укажи число дней от 1 до 730.')
            return
        expiry = grant(pupil_id,uid,days)
        await update.message.reply_text(f'✅ Доступ ученику {pupil_id} открыт до {expiry}.')
    elif command == '/revoke':
        deleted = revoke(pupil_id)
        await update.message.reply_text('✅ Доступ закрыт.' if deleted else 'Такого ID в списке не было.')


async def answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Preserve turn order within a user's chat while other students can use commands.
    lock = context.user_data.setdefault('_lock', asyncio.Lock())
    async with lock:
        await answer_locked(update, context)


async def answer_locked(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    yes, why = allowed(uid)
    if not yes:
        await update.message.reply_text(f'🔒 {why}\nТвой ID: {uid}. Передай его организатору для подтверждения доступа.')
        return
    question = update.message.text.strip()
    if len(question) < 2:
        await update.message.reply_text('Опиши, пожалуйста, что хочешь уточнить.')
        return
    if len(question) > 2000:
        await update.message.reply_text('Раздели, пожалуйста, вопрос на несколько сообщений до 2000 знаков.')
        return
    turns = load_turns(context.user_data)
    lookup = search_question(question, turns)
    try:
        hits = await retrieve(lookup)
    except (sqlite3.Error, OSError):
        LOG.warning('Не удалось прочитать базу знаний')
        await update.message.reply_text('Не удалось открыть базу знаний. Сообщи преподавателю, чтобы он проверил файлы бота.')
        return
    model_turns, max_chars = prompt_budget(question, turns, NUM_CTX, NUM_PREDICT)
    evidence = context_for(lookup, hits, max_chars=max_chars)
    if not evidence:
        await update.message.reply_text(NO_EVIDENCE)
        remember(context.user_data, question, NO_EVIDENCE, turns)
        return
    quota_day = now_utc().date().isoformat()
    if not consume(uid):
        await update.message.reply_text(f'Лимит {MAX_DAILY} вопросов на сегодня исчерпан. Можно продолжить завтра.')
        return
    if PROVIDER == 'offline':
        output = 'Режим поиска без ИИ: ниже непроверенные выдержки, а не готовая рекомендация.\n\n' + evidence
    else:
        typing = asyncio.create_task(keep_typing(context, update.effective_chat.id))
        try:
            await asyncio.wait_for(MODEL_GATE.acquire(), timeout=120)
            try:
                output = await produce_answer(question, evidence, model_turns, generate_response)
            finally:
                MODEL_GATE.release()
        except asyncio.CancelledError:
            refund(uid, quota_day)
            raise
        except Exception as exc:
            refund(uid, quota_day)
            LOG.warning('Ответ модели не получен (%s)', type(exc).__name__)
            await update.message.reply_text('Не удалось получить ответ. Попробуй ещё раз чуть позже. Преподавателю: проверь запуск Ollama и название установленной модели.')
            return
        finally:
            typing.cancel()
            with suppress(asyncio.CancelledError):
                await typing
    for part in split_telegram(output):
        await update.message.reply_text(part)
    # Raw excerpts are not assistant answers and should not pollute dialogue.
    if PROVIDER != 'offline':
        remember(context.user_data, question, output, turns)


async def other_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    yes, why = allowed(update.effective_user.id)
    if not yes:
        await update.message.reply_text(f'🔒 {why}')
    else:
        await update.message.reply_text('Пока я отвечаю только на текстовые вопросы. Опиши повреждение словами: цвет, материал, этап, что получилось и что ожидалось.')


def main():
    if not TOKEN:
        raise SystemExit('Заполни TELEGRAM_BOT_TOKEN в файле .env (см. README).')
    if PROVIDER == 'proxyapi' and not ALLOW_EXTERNAL:
        raise SystemExit('Внешний ИИ выключен. Для локальной работы выбери AI_PROVIDER=ollama. Передача переписки в ProxyAPI требует явного ALLOW_EXTERNAL_AI=true.')
    if PROVIDER == 'proxyapi' and not KEY:
        raise SystemExit('Заполни AI_API_KEY ключом ProxyAPI в файле .env. Или выбери AI_PROVIDER=offline.')
    if not ADMINS or any(not x.strip().isdigit() for x in ADMIN_ID_INPUT.split(',')):
        raise SystemExit('ADMIN_TELEGRAM_IDS: укажи свой ЧИСЛОВОЙ Telegram ID (только цифры, не ник). Узнать можно у @userinfobot.')
    if not KNOWLEDGE.is_file():
        raise SystemExit('Нет knowledge.sqlite рядом с bot.py. Распакуй архив целиком.')
    if not 4096 <= NUM_CTX <= 32768 or not 256 <= NUM_PREDICT <= 2048 or MAX_DAILY < 1:
        raise SystemExit('Проверь настройки: OLLAMA_NUM_CTX 4096–32768, OLLAMA_NUM_PREDICT 256–2048, дневной лимит больше нуля.')
    init_db()
    app = ApplicationBuilder().token(TOKEN).concurrent_updates(8).build()
    app.add_handler(CommandHandler('start',start))
    app.add_handler(CommandHandler('id',my_id))
    app.add_handler(CommandHandler('status',status))
    app.add_handler(CommandHandler('help',help_command))
    app.add_handler(CommandHandler(['new', 'reset'],new_topic))
    for cmd in ['grant','revoke','stats']:
        app.add_handler(CommandHandler(cmd,admin_command))
    app.add_handler(MessageHandler(filters.ChatType.PRIVATE & filters.TEXT & ~filters.COMMAND,answer))
    app.add_handler(MessageHandler(filters.ChatType.PRIVATE & ~filters.COMMAND,other_media))
    LOG.info('Бот запущен: режим %s; модель %s', PROVIDER, MODEL if PROVIDER != 'offline' else 'не нужна')
    async def on_error(update, context):
        LOG.warning('Ошибка обработки Telegram (%s)', type(context.error).__name__)
    app.add_error_handler(on_error)
    app.run_polling(drop_pending_updates=False)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    main()
