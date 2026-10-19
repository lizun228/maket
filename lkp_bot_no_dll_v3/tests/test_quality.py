import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from conversation import load_turns, remember, search_question, relevant_history
from response_quality import clean_answer, answer_issues, split_telegram, make_prompt, UNCERTAIN
from qa_pipeline import produce_answer
from local_models import ollama_url, chat
import httpx


class DialogueTests(unittest.TestCase):
    def test_new_explicit_topic_does_not_inherit_short_previous_question(self):
        turns=[('Почему выдирает каттером?', 'Проверь высыхание.')]
        q='Как подобрать серебристый металлик?'
        self.assertEqual(search_question(q,turns),q)
        self.assertEqual(relevant_history(q,turns),[])

    def test_short_follow_up_retains_topic_but_never_searches_previous_answer(self):
        old='Почему выдирает каттером?'
        turns=[(old,'Несуществующий материал Unicorn123')]
        query=search_question('А если без ИК?',turns)
        self.assertIn(old,query)
        self.assertNotIn('Unicorn123',query)

    def test_topic_survives_many_clarifications(self):
        data={}
        remember(data,'Почему выдирает каттером?','Ответ',[])
        for _ in range(8):
            remember(data,'А если без ИК?','Ответ',data['turns'])
        self.assertEqual(len(data['turns']),4)
        self.assertIn('каттером',search_question('А сколько ждать?',data['turns']))
        self.assertIn('каттером',make_prompt('А сколько ждать?','Сведения',data['turns']))

    def test_material_name_after_clarifying_question_keeps_original_topic(self):
        turns=[('Почему выдирает каттером?', 'Какой материал используешь?')]
        self.assertIn('каттером',search_question('Акриловый состав',turns))
        self.assertEqual(search_question('Как подобрать металлик?',turns),'Как подобрать металлик?')

    def test_expiry_and_history_bound(self):
        data={'turns':[('q','a')],'turns_at':100}
        self.assertEqual(load_turns(data,now=2000),[])


class OutputTests(unittest.TestCase):
    def test_reasoning_citations_and_markdown_removed_but_measurement_kept(self):
        text='<think>secret internal text</think>**Проверь материал.** [1] (источник 3) Температура 20 °C.'
        out=clean_answer(text)
        self.assertNotIn('secret',out)
        self.assertNotIn('источник',out)
        self.assertNotIn('[1]',out)
        self.assertIn('20 °C',out)

    def test_unclosed_thinking_not_shown(self):
        self.assertEqual(clean_answer('<think>Рассуждаю дальше'), '')

    def test_unsubstantiated_schedule_and_ratio_rejected(self):
        self.assertTrue(answer_issues('Суши 30 минут при 60 °C, смешай 2:1.', 'Материал бывает мягким.'))
        self.assertFalse(answer_issues('Подожди 10 мин.', 'Выдержка 10 минут.'))

    def test_split_respects_telegram_limit_and_keeps_all_words(self):
        text=('Первый полезный абзац.\n\n'+'Следующий важный совет. '*500).strip()
        parts=split_telegram(text)
        self.assertTrue(all(len(p)<=3500 for p in parts))
        self.assertEqual(' '.join(text.split()),' '.join(' '.join(parts).split()))


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_evidence_does_not_call_model(self):
        generate=AsyncMock()
        answer=await produce_answer('Почему выдирает?', '', [], generate)
        generate.assert_not_awaited()
        self.assertTrue(answer)

    async def test_unstructured_metatext_never_reaches_student(self):
        evidence='Если материал не просох, каттер может вырывать его при подрезке.'
        generate=AsyncMock(return_value='Ученик спрашивает о каттере. Проверь состав.')
        out=await produce_answer('Почему выдирает каттером?',evidence,[],generate)
        self.assertNotIn('Ученик спрашивает',out)
        self.assertEqual(generate.await_count,2)

    async def test_stripping_source_number_does_not_publish_model_rewrite(self):
        evidence='Если материал не просох, каттер может вырывать его при подрезке.'
        generate=AsyncMock(return_value='В сообщении 1234 советуют добавить неизвестный компонент.')
        out=await produce_answer('Почему выдирает каттером?',evidence,[],generate)
        self.assertNotIn('1234',out)
        self.assertNotIn('неизвестный компонент',out)
        self.assertEqual(generate.await_count,2)

    async def test_invented_parameters_rejected_without_rewriting(self):
        evidence='Если материал не просох, каттер может вырывать его при подрезке.'
        generate=AsyncMock(return_value='Суши 999 минут при 777 °C.')
        out=await produce_answer('Почему выдирает каттером?',evidence,[],generate)
        self.assertNotIn('999',out)
        self.assertNotIn('777',out)
        self.assertEqual(generate.await_count,2)

    async def test_ollama_payload_is_local_nonstreaming_and_disables_thinking(self):
        seen=[]
        def handler(request):
            seen.append((request,json.loads(request.content)))
            return httpx.Response(200,json={'message':{'content':'Проверь высыхание.'}})
        original=httpx.AsyncClient
        def client(**kwargs):
            self.assertFalse(kwargs['trust_env'])
            self.assertFalse(kwargs['follow_redirects'])
            return original(**kwargs,transport=httpx.MockTransport(handler))
        with patch('local_models.httpx.AsyncClient',side_effect=client):
            out=await chat('http://127.0.0.1:11434','qwen3:4b-instruct',[{'role':'user','content':'Вопрос'}])
        request,payload=seen[0]
        self.assertEqual(request.url.host,'127.0.0.1')
        self.assertEqual(request.url.path,'/api/chat')
        self.assertFalse(payload['think'])
        self.assertFalse(payload['stream'])
        self.assertEqual(payload['options']['num_ctx'],8192)
        self.assertEqual(out,'Проверь высыхание.')

    async def test_old_qwen_thinking_field_compatibility_retry(self):
        seen=[]
        def handler(request):
            payload=json.loads(request.content);seen.append(payload)
            if len(seen)==1:
                return httpx.Response(400,json={'error':'model does not support thinking'})
            return httpx.Response(200,json={'message':{'content':'Ответ'}})
        original=httpx.AsyncClient
        with patch('local_models.httpx.AsyncClient',side_effect=lambda **kw: original(**kw,transport=httpx.MockTransport(handler))):
            self.assertEqual(await chat('http://localhost:11434','qwen3:4b-instruct',[]),'Ответ')
        self.assertNotIn('think',seen[1])


class PrivacyTests(unittest.TestCase):
    def test_remote_ollama_destinations_and_credentials_rejected(self):
        for base in ['https://example.com','http://127.0.0.1.evil.com','http://user:pass@localhost:11434','http://localhost:11434/path','http://localhost:11434/?x=1']:
            with self.subTest(base=base),self.assertRaises(ValueError):
                ollama_url(base,'/api/chat')
        self.assertEqual(ollama_url('http://127.0.0.1:11434/','/api/chat'),'http://127.0.0.1:11434/api/chat')
