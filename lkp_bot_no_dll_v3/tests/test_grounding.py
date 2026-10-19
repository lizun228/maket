"""Regression tests use synthetic source text, not uploaded private chat excerpts."""
import asyncio
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grounded_answers import (build_evidence,parse_selection,parse_review,
                              correction_response,policy_issue,SELECT_SYSTEM,REVIEW_SYSTEM)
from qa_pipeline import produce_answer
from response_quality import make_prompt
from search_engine import search_db,message_parts
import httpx
from local_models import chat

ROOT=Path(__file__).resolve().parents[1]
Q='состав который я наношу с кисточки долго сохнет'
# Explicitly synthetic test sources, never installed as application knowledge.
SOURCE='Если состав сохнет медленно, проверь условия сушки и толщину нанесённого слоя.'
CHECK='При проверке сравни поведение того же состава на отдельной тестовой поверхности.'


def evidence(text=SOURCE,question=Q):
    return build_evidence(question,[{'id':1,'content':'№101: '+text}])


def plan(unit,role='possible_reason'):
    return json.dumps({'status':'answer','selections':[{'id':unit.id,'role':role}]})


def review(unit,status='ok'):
    return json.dumps({'status':status,'approved_ids':[unit.id] if status=='ok' else []})


class SourceSelectionTests(unittest.TestCase):
    def test_candidate_is_complete_original_sentence_with_negation(self):
        source='Если состав не просох, не увеличивай усилие при подрезке каттером.'
        bundle=evidence(source,'Почему выдирает каттером?')
        self.assertTrue(bundle.units)
        self.assertEqual(bundle.units[0].text,source)
        self.assertIn('не увеличивай',bundle.units[0].text)

    def test_question_is_not_fact_and_adjacent_words_are_not_new_compound(self):
        source='Что за акриловый полимер?\nДля растворителя в этом вопросе употреблено слово акриловый.'
        bundle=evidence(source,'Что за акриловый полимер?')
        self.assertTrue(all('?' not in u.text for u in bundle.units))
        answer=correction_response('Что за акриловый полимер?',[],bundle)
        self.assertIsNotNone(answer)
        self.assertNotIn('базовый компонент',answer)

    def test_aerograph_scope_not_transferred_to_brush(self):
        source='Промывочный растворитель в аэрографе может менять поведение состава при сушке.'
        self.assertFalse(evidence(source).units)
        self.assertEqual(policy_issue(Q,source,source),'different_application_method')

    def test_uncertain_additive_claim_is_not_default_diagnosis(self):
        source='Если не добавлен разбавитель, материал в этом случае сохнет медленно.'
        self.assertFalse(evidence(source).units)

    def test_numeric_recipe_and_unknown_substrate_filtered_even_if_in_source(self):
        for source in ['Для сушки прогрей состав лампой 6-7 раз и приступай к подрезке.',
                       'На пластике данный состав сохнет дольше, чем ожидалось.',
                       'Добавляй больше разбавителя в состав, если он долго сохнет.']:
            with self.subTest(source=source): self.assertFalse(evidence(source).units)

    def test_material_chemistry_not_assumed_even_if_someone_mentions_it(self):
        source='Акриловый полимер в описанном случае сохнет медленно и требует отдельной проверки.'
        self.assertFalse(evidence(source).units)
        self.assertEqual(policy_issue(Q,source,source),'unknown_material_chemistry')

    def test_half_message_tail_not_reconstructed(self):
        beginning='Разговор о не относящемся к вопросу случае. '
        tail=' Состав сохнет дольше потому что неизвестное'
        source=(beginning*20)[:500-len(tail)]+tail
        self.assertEqual(len(source),500)
        bundle=evidence(source)
        self.assertFalse(any('неизвестное' in u.text for u in bundle.units))

    def test_selection_cannot_inject_quote_answer_or_unknown_id(self):
        bundle=evidence();unit=bundle.units[0]
        attacks=[{'status':'answer','selections':[{'id':'u999','role':'check'}]},
                 {'status':'answer','selections':[{'id':unit.id,'role':'check','text':'добавь растворитель'}]},
                 {'status':'answer','selections':[{'id':unit.id,'role':'check'}],'answer':'новая химия'},
                 {'status':'answer','selections':[{'id':unit.id,'role':['check']}]}]
        for value in attacks:
            with self.subTest(value=value): self.assertIsNone(parse_selection(json.dumps(value),bundle))

    def test_review_cannot_approve_unselected_quote(self):
        bundle=evidence();unit=bundle.units[0]
        self.assertIsNone(parse_review('{"status":"ok","approved_ids":["u999"]}',[(unit,'check')]))

    def test_previous_generated_facts_are_not_in_prompt(self):
        turns=[('Состав долго сохнет','Выдуманный акриловый полимер — это однокомпонентная основа.')]
        prompt=make_prompt('А если без лампы?',SOURCE,turns)
        self.assertNotIn('однокомпонентная основа',prompt)
        self.assertNotIn('предыдущий_ответ',prompt)

    def test_actual_archive_retrieval_reaches_relevant_reply_and_quotes_are_real(self):
        # Only assertions/IDs are committed. Actual private texts stay in the DB.
        hits=search_db(ROOT/'knowledge.sqlite',Q,6)
        self.assertIn(5256,[h['id'] for h in hits])
        bundle=build_evidence(Q,hits)
        self.assertTrue(any(u.message_id=='37187' for u in bundle.units))
        bodies={mid:body for h in hits for mid,body in message_parts(h['content'])}
        for unit in bundle.units:
            original=' '.join(bodies[unit.message_id].split())
            self.assertIn(' '.join(unit.text.split()),original)
            self.assertNotIn('акриловый полимер',unit.text.lower())
            self.assertIsNone(policy_issue(Q,unit.text,unit.context))


class GroundedPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_output_is_source_text_not_free_model_answer(self):
        bundle=evidence();unit=bundle.units[0]
        generate=AsyncMock(side_effect=[plan(unit),review(unit)])
        audit={}
        out=await produce_answer(Q,bundle,[],generate,audit=audit)
        self.assertIn(unit.text,out)
        self.assertEqual(audit['status'],'source_sentences')
        self.assertEqual(audit['selected'][0]['message_id'],'101')
        self.assertNotIn('№101',out)
        self.assertEqual(generate.await_count,2)
        self.assertEqual(generate.await_args_list[0].args[0],SELECT_SYSTEM)
        self.assertEqual(generate.await_args_list[1].args[0],REVIEW_SYSTEM)

    async def test_bad_answer_from_user_report_is_never_forwarded(self):
        bad='Это из-за недостатка разбавителя. Акриловый полимер — однокомпонентная основа.'
        generate=AsyncMock(return_value=bad)
        audit={}
        out=await produce_answer(Q,evidence(),[],generate,audit=audit)
        self.assertNotIn('Акриловый полимер',out)
        self.assertNotIn('недостатка разбавителя',out)
        self.assertEqual(audit['status'],'invalid_selection')
        self.assertEqual(generate.await_count,1)

    async def test_unknown_term_from_previous_error_is_corrected_not_defended(self):
        turns=[(Q,'Если используешь акриловый полимер, выбирай растворитель.')]
        generate=AsyncMock()
        audit={}
        out=await produce_answer('Что за акриловый полимер?',evidence(),turns,generate,audit=audit)
        self.assertIn('моя ошибка',out)
        self.assertNotIn('базовый компонент',out)
        self.assertNotIn('однокомпонент',out)
        self.assertEqual(audit['status'],'unsupported_term')
        generate.assert_not_awaited()

    async def test_explicit_material_correction_in_other_case_is_not_defended(self):
        turns=[(Q,'Если используешь акриловый полимер, выбирай растворитель.')]
        generate=AsyncMock()
        from grounded_answers import Evidence,Unit
        source='Акриловый полимер упоминается в другом описании, но это не определяет состав ученика.'
        bundle=Evidence((Unit('u1',source,source,1,'101'),),Q)
        out=await produce_answer('Никакого акрилового полимера у нас нет',bundle,turns,generate)
        self.assertIn('моя ошибка',out)
        self.assertNotIn('однокомпонент',out)
        generate.assert_not_awaited()

    async def test_reviewer_rejection_closes_response_without_showing_quote(self):
        bundle=evidence();unit=bundle.units[0]
        generate=AsyncMock(side_effect=[plan(unit),review(unit,'insufficient')])
        out=await produce_answer(Q,bundle,[],generate)
        self.assertNotIn(unit.text,out)
        self.assertIn('Как сушишь',out)

    async def test_malformed_review_cannot_leak_prose(self):
        bundle=evidence();unit=bundle.units[0]
        generate=AsyncMock(side_effect=[plan(unit),'Добавь неизвестный компонент в смесь.'])
        out=await produce_answer(Q,bundle,[],generate)
        self.assertNotIn('неизвестный компонент',out)
        self.assertNotIn(unit.text,out)

    async def test_model_history_contains_user_questions_only(self):
        bundle=evidence();unit=bundle.units[0]
        turns=[(Q,'Секретная выдумка модели UnicornChemical')]
        generate=AsyncMock(side_effect=[plan(unit),review(unit)])
        await produce_answer('А если без лампы?',bundle,turns,generate)
        for call in generate.await_args_list:
            self.assertNotIn('UnicornChemical',call.args[1])
            self.assertIn(Q,call.args[1])

    async def test_empty_evidence_does_not_call_model(self):
        generate=AsyncMock()
        await produce_answer(Q,build_evidence(Q,[]),[],generate)
        generate.assert_not_awaited()

    async def test_json_format_and_zero_temperature_are_sent_to_ollama(self):
        seen=[]
        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200,json={'message':{'content':'{"status":"insufficient","selections":[]}'}})
        original=httpx.AsyncClient
        with patch('local_models.httpx.AsyncClient',side_effect=lambda **kw: original(**kw,transport=httpx.MockTransport(handler))):
            await chat('http://127.0.0.1:11434','qwen3:4b-instruct',[],json_mode=True)
        self.assertEqual(seen[0]['format'],'json')
        self.assertEqual(seen[0]['options']['temperature'],0)
        self.assertFalse(seen[0]['think'])


class BotGroundingIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import tempfile,bot
        self.bot=bot
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        for name,value in [('ACCESS_DB',Path(self.tmp.name)/'access.sqlite'),('ADMINS',{999}),('PROVIDER','ollama'),('MODEL_GATE',asyncio.Semaphore(1))]:
            item=patch.object(bot,name,value);item.start();self.addCleanup(item.stop)
        bot.init_db();bot.grant(123,999,1)
        from types import SimpleNamespace
        self.message=SimpleNamespace(text=Q,reply_text=AsyncMock())
        self.update=SimpleNamespace(effective_user=SimpleNamespace(id=123),effective_chat=SimpleNamespace(id=123,type='private'),message=self.message)
        self.context=SimpleNamespace(user_data={},bot=SimpleNamespace(send_chat_action=AsyncMock()))
        self.hits=[{'id':1,'content':'№101: '+SOURCE}]

    async def test_full_bot_path_sends_only_audited_quote_and_keeps_quota(self):
        async def generate(system,prompt):
            payload=json.loads(prompt)
            if 'available_units' in payload:
                return json.dumps({'status':'answer','selections':[{'id':payload['available_units'][0]['id'],'role':'possible_reason'}]})
            return json.dumps({'status':'ok','approved_ids':[payload['selected'][0]['id']]})
        with patch.object(self.bot,'retrieve',AsyncMock(return_value=self.hits)),patch.object(self.bot,'generate_response',side_effect=generate):
            await self.bot.answer(self.update,self.context)
        out=self.message.reply_text.await_args.args[0]
        self.assertIn(SOURCE,out)
        self.assertNotIn('№101',out)
        self.assertEqual(self.context.user_data['grounding']['status'],'source_sentences')
        with self.bot.db() as con:
            self.assertEqual(con.execute('SELECT count FROM usage WHERE user_id=123').fetchone()[0],1)

    async def test_invalid_model_output_refunds_request(self):
        with patch.object(self.bot,'retrieve',AsyncMock(return_value=self.hits)),patch.object(self.bot,'generate_response',AsyncMock(return_value='Выдуманное объяснение вместо выбора.')):
            await self.bot.answer(self.update,self.context)
        with self.bot.db() as con:
            self.assertEqual(con.execute('SELECT count FROM usage WHERE user_id=123').fetchone()[0],0)
        self.assertNotIn('Выдуманное',self.message.reply_text.await_args.args[0])

    async def test_evidence_command_cannot_disclose_quotes_in_a_group(self):
        self.update.effective_user.id=999
        self.update.effective_chat.type='group'
        self.context.user_data['grounding']={'selected':[{'text':'private quote','message_id':'1','passage_id':1}]}
        await self.bot.evidence_command(self.update,self.context)
        self.message.reply_text.assert_not_awaited()
