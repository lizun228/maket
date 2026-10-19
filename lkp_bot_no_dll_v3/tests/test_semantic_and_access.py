from pathlib import Path
import asyncio
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bot
from build_index import build
from semantic_search import build_vectors, nearest, _load
from response_quality import prompt_budget


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.src=self.root/'source.jsonl';self.db=self.root/'knowledge.sqlite';self.cache=self.root/'embeddings.sqlite'
        rows=[{'thread_id':'a','messages':[{'message_id':1,'text':'Каттер вырывает полимер. Проверь высыхание.'}]},
              {'thread_id':'b','messages':[{'message_id':2,'text':'Серебристый металлик сложно наносить кистью.'}]}]
        self.src.write_text('\n'.join(json.dumps(row,ensure_ascii=False) for row in rows),encoding='utf-8')
        build(self.src,self.db)
        self.addCleanup(_load.cache_clear)

    def vectors(self,base,model,texts):
        return [[1.,0.] if ('Каттер' in text or 'выдирает' in text) else [0.,1.] for text in texts]

    def test_local_vector_cache_ranking_and_source_invalidation(self):
        with patch('semantic_search.model_digest',return_value='digest1'),patch('semantic_search.embed',side_effect=self.vectors):
            self.assertEqual(build_vectors(self.db,self.cache,'http://localhost:11434','bge-m3'),2)
            result=nearest(self.db,self.cache,'выдирает','http://localhost:11434','bge-m3')
            self.assertEqual(result[0][0],1)
            self.assertAlmostEqual(result[0][1],1)
            with sqlite3.connect(self.db) as c: c.execute("INSERT INTO metadata VALUES('test_change','yes')")
            with self.assertRaisesRegex(ValueError,'устарел'):
                nearest(self.db,self.cache,'выдирает','http://localhost:11434','bge-m3')

    def test_model_change_invalidates_vectors(self):
        with patch('semantic_search.model_digest',return_value='digest1'),patch('semantic_search.embed',side_effect=self.vectors):
            build_vectors(self.db,self.cache,'http://localhost:11434','bge-m3')
        with patch('semantic_search.model_digest',return_value='digest2'),patch('semantic_search.embed') as embedding:
            with self.assertRaisesRegex(ValueError,'изменилась'):
                nearest(self.db,self.cache,'вопрос','http://localhost:11434','bge-m3')
            embedding.assert_not_called()

    def test_vector_builder_cannot_replace_archive_or_access_db(self):
        for target in [self.db,self.root/'access.sqlite']:
            if not target.exists():
                with sqlite3.connect(target) as c: c.execute('CREATE TABLE access(user_id INTEGER)')
            before=target.read_bytes()
            with self.assertRaises(ValueError): build_vectors(self.db,target,'http://localhost:11434','bge-m3',replace=True)
            self.assertEqual(target.read_bytes(),before)

    def test_knowledge_builder_cannot_replace_access_db(self):
        target=self.root/'access.sqlite'
        with sqlite3.connect(target) as c: c.execute('CREATE TABLE access(user_id INTEGER)')
        before=target.read_bytes()
        with self.assertRaises(ValueError): build(self.src,target,replace=True)
        self.assertEqual(target.read_bytes(),before)

    def test_prompt_budget_reduces_history_and_context_for_smaller_windows(self):
        turns=[('Почему выдирает каттером? '+'текст '*100,'ответ '*200)]*4
        _, small=prompt_budget('А если без ИК?',turns,4096,650)
        _, large=prompt_budget('А если без ИК?',turns,8192,650)
        self.assertLess(small,large)
        self.assertGreater(small,0)


class AccessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.patcher=patch.object(bot,'ACCESS_DB',Path(self.tmp.name)/'access.sqlite');self.patcher.start();self.addCleanup(self.patcher.stop)
        self.admins=patch.object(bot,'ADMINS',{999});self.admins.start();self.addCleanup(self.admins.stop)
        bot.init_db()

    async def test_existing_grants_and_limits_keep_working(self):
        with patch.object(bot,'MAX_DAILY',1):
            self.assertFalse(bot.allowed(123)[0])
            bot.grant(123,999,1)
            self.assertTrue(bot.allowed(123)[0])
            self.assertTrue(bot.consume(123))
            self.assertFalse(bot.consume(123))
            bot.refund(123)
            self.assertTrue(bot.consume(123))
            self.assertTrue(bot.revoke(123))
            self.assertFalse(bot.allowed(123)[0])

    async def test_model_error_refunds_quota_and_preserves_history(self):
        from types import SimpleNamespace
        bot.grant(123,999,1)
        message=SimpleNamespace(text='Почему выдирает каттером?',reply_text=AsyncMock())
        update=SimpleNamespace(effective_user=SimpleNamespace(id=123),effective_chat=SimpleNamespace(id=123),message=message)
        context=SimpleNamespace(user_data={},bot=SimpleNamespace(send_chat_action=AsyncMock()))
        hits=[{'content':'№1: Каттер выдирает полимер?\n№2: Если полимер не просох, проверь высыхание.'}]
        with patch.object(bot,'retrieve',AsyncMock(return_value=hits)),patch.object(bot,'generate_response',AsyncMock(side_effect=RuntimeError('private data'))),patch.object(bot,'MODEL_GATE',asyncio.Semaphore(1)):
            await bot.answer(update,context)
        with bot.db() as c:
            self.assertEqual(c.execute('SELECT count FROM usage WHERE user_id=123').fetchone()[0],0)
        self.assertNotIn('turns',context.user_data)
        self.assertNotIn('private data',message.reply_text.await_args.args[0])

    async def test_external_ai_requires_explicit_consent_before_request(self):
        with patch.object(bot,'PROVIDER','proxyapi'),patch.object(bot,'ALLOW_EXTERNAL',False),patch('bot.httpx.AsyncClient') as client:
            with self.assertRaises(ValueError): await bot.generate_response('system','private archive')
            client.assert_not_called()
