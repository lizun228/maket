from pathlib import Path
import hashlib
import json
import sqlite3
import tempfile
import unittest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_index import build, clean_message
from search_engine import search_db, context_for, query_terms


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.source=self.root/'source.jsonl'
        self.target=self.root/'knowledge.sqlite'
        self.write_thread([
            {'message_id':1,'text':'Почему каттер выдирает полимер из скола?'},
            {'message_id':2,'text':'Если полимер не просох, не торопись подрезать. Проверь высыхание и технику срезания.'},
        ])

    def write_thread(self,messages):
        self.source.write_text(json.dumps({'thread_id':'t1','messages':messages},ensure_ascii=False)+'\n',encoding='utf-8')

    def test_build_search_and_context_have_answer_without_source_ids(self):
        build(self.source,self.target)
        hits=search_db(self.target,'Почему вырывает при подрезке каттером?')
        self.assertTrue(hits)
        context=context_for('Почему вырывает при подрезке каттером?',hits)
        self.assertIn('Проверь высыхание',context)
        self.assertNotIn('№',context)
        self.assertNotIn('Источник',context)

    def test_unsupported_question_is_not_just_any_shared_word(self):
        build(self.source,self.target)
        self.assertEqual(search_db(self.target,'Какое давление в шинах зимой?'),[])

    def test_no_overwrite_without_flag_and_failure_preserves_old_index(self):
        build(self.source,self.target)
        before=self.target.read_bytes()
        with self.assertRaises(FileExistsError): build(self.source,self.target)
        self.source.write_text('{invalid JSON',encoding='utf-8')
        with self.assertRaises(json.JSONDecodeError): build(self.source,self.target,replace=True)
        self.assertEqual(self.target.read_bytes(),before)
        self.assertEqual(list(self.root.glob('*.partial')),[])

    def test_successful_replace_preserves_verified_previous_database(self):
        build(self.source,self.target)
        previous=self.target.read_bytes()
        self.write_thread([{'message_id':3,'text':'Полимер не просох, каттер может вырывать материал.'}])
        build(self.source,self.target,replace=True)
        backups=list(self.root.glob('*.bak'))
        self.assertEqual(len(backups),1)
        with sqlite3.connect(backups[0]) as c:
            self.assertEqual(c.execute('SELECT count(*) FROM passages').fetchone()[0],1)
            self.assertIn('Почему каттер',c.execute('SELECT content FROM passages').fetchone()[0])
        self.assertNotEqual(previous,self.target.read_bytes())

    def test_long_message_not_cut_to_500_and_context_reaches_late_answer(self):
        body='Обсуждали организацию сервиса. '*100+'Если каттер выдирает полимер, проверь высыхание. Не усиливай нажим.'
        self.write_thread([{'message_id':1,'text':body}])
        build(self.source,self.target)
        hits=search_db(self.target,'Почему каттер выдирает полимер?')
        self.assertIn('Не усиливай нажим',hits[0]['content'])
        context=context_for('Почему каттер выдирает полимер?',hits,max_chars=1000)
        self.assertIn('проверь высыхание',context)
        self.assertLessEqual(len(context),1000)

    def test_full_telegram_rich_text_and_reply_across_pause(self):
        source=self.root/'result.json'
        source.write_text(json.dumps({'messages':[
            {'id':1,'type':'message','date_unixtime':'100','text':['Почему каттер ',{'type':'bold','text':'выдирает полимер?'}]},
            {'id':2,'type':'service','text':'Служебная запись'},
            {'id':3,'type':'message','date_unixtime':'10000','reply_to_message_id':1,'text':'Если полимер не просох, при подрезке каттер вырывает его. Проверь высыхание.'}
        ]},ensure_ascii=False),encoding='utf-8')
        build(source,self.target)
        with sqlite3.connect(self.target) as c:
            rows=[r[0] for r in c.execute('SELECT content FROM passages')]
        self.assertTrue(any('№1:' in r and '№3:' in r for r in rows))
        self.assertFalse(any('Служебная запись' in r for r in rows))

    def test_overlapping_messages_sent_to_model_once(self):
        content='№1: Почему каттер выдирает материал?\n№2: Проверь высыхание материала перед подрезкой.'
        hits=[{'content':content},{'content':content+'\n№3: Не усиливай нажим каттера.'}]
        context=context_for('Каттер выдирает',hits)
        self.assertEqual(context.count('Проверь высыхание'),1)

    def test_ik_not_dropped_and_withdraw_not_confused_with_hold_time(self):
        self.assertIn('ик',query_terms('Почему просадка после ИК?'))
        self.assertNotIn('отрыв',query_terms('Выдержка между слоями'))
