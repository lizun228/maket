"""Synthetic model responses test contracts, not the intelligence of real Ollama."""
import asyncio
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grounded_answers import build_evidence,correction_response,policy_issue,Evidence,Unit
from answer_composer import parse_draft,parse_verdict,COMPOSE_SYSTEM,VERIFY_SYSTEM,available_questions
from conversation import build_case,remember,load_turns,extract_facts
from qa_pipeline import produce_answer
from response_quality import make_prompt
from search_engine import search_db,message_parts
import httpx
from local_models import chat

ROOT=Path(__file__).resolve().parents[1]
Q='состав который я наношу кисточкой долго сохнет'
SOURCE='Если состав сохнет медленно, проверь условия сушки и толщину нанесённого слоя.'
IR_SOURCE='При ИК-сушке толстый слой состава может сохнуть дольше тонкого.'
PARAPHRASE='Долгая сушка может быть связана с толщиной слоя или условиями сушки.'


def evidence(text=SOURCE,question=Q):
    return build_evidence(question,[{'id':1,'content':'№101: '+text}])


def claim(text=PARAPHRASE,source_ids=None,kind='possible_reason',facts=None,ident='c1'):
    return {'id':ident,'kind':kind,'text':text,'source_ids':['u1'] if source_ids is None else source_ids,'fact_keys':facts or []}


def draft(claims=None,followup=None):
    return json.dumps({'status':'answer','claims':[claim()] if claims is None else claims,'followup':followup},ensure_ascii=False)


def verdict(claims=None,ok=True,dialogue=True):
    return json.dumps({'claims':[{'id':c['id'],'supported':ok,'reason':'synthetic check'} for c in (claims or [claim()])],
                       'dialogue_ok':dialogue,'reason':'synthetic dialogue check'})


class EvidenceTests(unittest.TestCase):
    def test_source_keeps_conditions_and_negation(self):
        source='Если состав не просох, не увеличивай усилие при подрезке каттером.'
        bundle=evidence(source,'Почему выдирает каттером?')
        self.assertEqual(bundle.units[0].text,source)

    def test_question_does_not_prove_compound_material_name(self):
        bundle=evidence('Что за акриловый полимер?\nДля растворителя в этом вопросе употреблено слово акриловый.','Что за акриловый полимер?')
        self.assertTrue(all('?' not in u.text for u in bundle.units))
        self.assertIsNotNone(correction_response('Что за акриловый полимер?',[],bundle))

    def test_aerograph_scope_not_transferred_to_brush(self):
        source='Промывочный растворитель в аэрографе может менять поведение состава при сушке.'
        self.assertFalse(evidence(source).units)
        self.assertEqual(policy_issue(Q,source,source),'different_application_method')

    def test_unknown_additive_not_default_diagnosis(self):
        self.assertFalse(evidence('Если не добавлен разбавитель, материал в этом случае сохнет медленно.').units)

    def test_unknown_recipe_regime_and_substrate_filtered(self):
        for source in ['Для сушки прогрей состав лампой 6-7 раз и приступай к подрезке.',
                       'На пластике данный состав сохнет дольше, чем ожидалось.',
                       'Добавляй больше разбавителя в состав, если он долго сохнет.']:
            with self.subTest(source=source): self.assertFalse(evidence(source).units)

    def test_numeric_observation_kept_as_context_not_automatically_prescribed(self):
        source='У меня этот состав оставался мягким после 2 минут ИК-сушки.'
        self.assertTrue(evidence(source).units)

    def test_unknown_chemistry_not_assumed(self):
        self.assertFalse(evidence('Акриловый полимер в описанном случае сохнет медленно и требует отдельной проверки.').units)

    def test_truncated_message_tail_not_reconstructed(self):
        tail=' Состав сохнет дольше потому что неизвестное'
        source=('Разговор о другом случае. '*30)[:500-len(tail)]+tail
        self.assertEqual(len(source),500)
        self.assertFalse(any('неизвестное' in u.text for u in evidence(source).units))

    def test_actual_archive_provenance_and_relevant_reply(self):
        hits=search_db(ROOT/'knowledge.sqlite',Q,6)
        self.assertIn(5256,[h['id'] for h in hits])
        bundle=build_evidence(Q,hits)
        self.assertTrue(any(u.message_id=='37187' for u in bundle.units))
        bodies={mid:body for h in hits for mid,body in message_parts(h['content'])}
        for unit in bundle.units:
            self.assertIn(' '.join(unit.text.split()),' '.join(bodies[unit.message_id].split()))
            self.assertIsNone(policy_issue(Q,unit.text,unit.context))


class DialogueStateTests(unittest.TestCase):
    def setUp(self):
        self.first=build_case(Q,[])
        self.turns=[(Q,'Как сушишь состав — естественно или лампой?')]
        self.second=build_case('сушу лампой ик по 2 минуты',self.turns,self.first)

    def test_reported_followup_retains_brush_and_distinguishes_cycle_from_total(self):
        facts=self.second['facts']
        self.assertTrue(self.second['is_followup'])
        self.assertEqual(facts['application_method']['value'],'brush')
        self.assertEqual(facts['drying_method']['value'],'infrared')
        self.assertEqual(facts['cycle_duration']['value'],'2')
        self.assertNotIn('total_elapsed',facts)
        both=extract_facts('Сушу ИК по 2 минуты, всего прошло 10 минут')
        self.assertEqual(both['cycle_duration']['value'],'2')
        self.assertEqual(both['total_elapsed']['value'],'10')
        self.assertNotIn('drying_method',available_questions(self.second))
        self.assertIn('cycle_count',available_questions(self.second))

    def test_correction_supersedes_obsolete_fact(self):
        c=build_case('нет, сушу естественно без лампы',[(self.second['latest_message'],'Сколько циклов?')],self.second)
        self.assertEqual(c['facts']['drying_method']['value'],'natural')
        self.assertIn('drying_method',c['changed_facts'])
        self.assertNotIn('cycle_count',available_questions(c))

    def test_new_topic_and_expiry_clear_case(self):
        c=build_case('Как подобрать серебристый металлик?',self.turns,self.second)
        self.assertFalse(c['is_followup'])
        self.assertNotIn('drying_method',c['facts'])
        data={'case':self.second,'turns':self.turns,'turns_at':100}
        load_turns(data,now=2000)
        self.assertNotIn('case',data)

    def test_hypothesis_not_saved_as_actual_observation(self):
        self.assertEqual(extract_facts('А если сушить ИК по 5 минут?'),{})

    def test_two_declarative_followups_keep_the_same_case(self):
        c=build_case('сделал 4 цикла, материал всё ещё мягкий',
                     [(Q,'Ответ'),(self.second['latest_message'],'Ответ')],self.second)
        self.assertEqual(c['topic'],Q)
        self.assertEqual(c['facts']['cycle_duration']['value'],'2')
        self.assertEqual(c['facts']['cycle_count']['value'],'4')

    def test_state_negations_and_multiple_observations_preserved(self):
        f=extract_facts('Не липнет, но материал всё ещё мягкий')
        self.assertIn('не липнет',f['material_state']['value'])
        self.assertIn('мягкий',f['material_state']['value'])

    def test_correction_does_not_reintroduce_obsolete_application_later(self):
        c=build_case('нет, наношу не кистью, а аэрографом',self.turns,self.first)
        c2=build_case('А что ещё проверить?',[(c['latest_message'],'Ответ')],c)
        self.assertEqual(c2['facts']['application_method']['value'],'airbrush')
        self.assertNotIn('кист',c2['search_query'])

    def test_fact_history_survives_many_turns(self):
        data={'case':self.second}
        remember(data,self.second['latest_message'],'Сколько таких циклов нагрева уже сделал?',self.turns)
        for _ in range(8):
            remember(data,'А что ещё проверить?','Ответ',data['turns'])
        self.assertEqual(data['case']['facts']['cycle_duration']['value'],'2')
        self.assertEqual(data['case']['topic'],Q)

    def test_latest_clarification_changes_real_retrieval(self):
        before=search_db(ROOT/'knowledge.sqlite',self.first['search_query'],6)
        after=search_db(ROOT/'knowledge.sqlite',self.second['search_query'],6,focus_query=self.second['focus_query'])
        self.assertNotEqual([h['id'] for h in before],[h['id'] for h in after])
        self.assertIn('ик',after[0]['matched_terms'])
        self.assertIn('сушка',after[0]['matched_terms'])
        self.assertIn(5256,[h['id'] for h in after])


class DraftContractTests(unittest.TestCase):
    def setUp(self):
        self.bundle=evidence();self.case=build_case(Q,[])

    def test_paraphrase_accepted_with_valid_provenance(self):
        value,issue=parse_draft(draft(),self.bundle,self.case)
        self.assertIsNone(issue)
        self.assertNotEqual(value['claims'][0]['text'],self.bundle.units[0].text)

    def test_long_verbatim_quote_requests_rephrasing(self):
        self.assertIsNone(parse_draft(draft([claim(SOURCE)]),self.bundle,self.case)[0])

    def test_unknown_source_and_technical_claim_without_source_rejected(self):
        for c in [claim(source_ids=['u999']),claim(source_ids=[])]:
            self.assertIsNone(parse_draft(draft([c]),self.bundle,self.case)[0])

    def test_invented_numbers_and_combined_chemistry_rejected(self):
        for text in ['Суши материал 999 минут при 777 °C.',
                     'Для сушки этого состава нужно 999 циклов нагрева.',
                     'Акриловый полимер представляет собой однокомпонентную основу.']:
            self.assertIsNone(parse_draft(draft([claim(text)]),self.bundle,self.case)[0])

    def test_review_requires_every_claim_and_real_boolean(self):
        d=json.loads(draft())
        self.assertIsNone(parse_verdict('{"claims":[],"dialogue_ok":true,"reason":"ok"}',d))
        self.assertIsNone(parse_verdict('{"claims":[{"id":"c1","supported":"true","reason":"ok"}],"dialogue_ok":true,"reason":"ok"}',d))

    def test_answer_cannot_ask_known_method_or_ignore_latest_facts(self):
        c=build_case('сушу лампой ик по 2 минуты',[(Q,'Как сушишь?')],self.case)
        self.assertIsNone(parse_draft(draft(followup='drying_method'),self.bundle,c)[0])
        self.assertEqual(parse_draft(draft(),self.bundle,c)[1],'latest_facts_ignored')
        # Metadata alone is insufficient: the answer must address the new facts.
        fake=claim(facts=['drying_method','cycle_duration'])
        self.assertIsNone(parse_draft(draft([fake]),self.bundle,c)[0])

    def test_previous_invented_answer_not_used_as_prompt_evidence(self):
        turns=[('Состав долго сохнет','Выдуманная однокомпонентная основа UnicornChemical')]
        self.assertNotIn('UnicornChemical',make_prompt('А если без лампы?',SOURCE,turns))


class SynthesisPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_paraphrase_sent_after_review_and_sources_only_in_audit(self):
        generate=AsyncMock(side_effect=[draft(),verdict()]);audit={}
        out=await produce_answer(Q,evidence(),[],generate,audit=audit)
        self.assertEqual(out,PARAPHRASE)
        self.assertNotIn(SOURCE,out)
        self.assertEqual(audit['status'],'grounded_synthesis')
        self.assertEqual(audit['selected'][0]['message_id'],'101')
        self.assertEqual(generate.await_args_list[0].args[0],COMPOSE_SYSTEM)
        self.assertEqual(generate.await_args_list[1].args[0],VERIFY_SYSTEM)

    async def test_unstructured_bad_answer_never_forwarded_and_retry_bounded(self):
        bad='Это из-за недостатка разбавителя. Акриловый полимер — однокомпонентная основа.'
        generate=AsyncMock(return_value=bad);audit={}
        out=await produce_answer(Q,evidence(),[],generate,audit=audit)
        self.assertNotIn('Акриловый полимер',out)
        self.assertEqual(audit['status'],'unconfirmed')
        self.assertEqual(generate.await_count,2)

    async def test_semantic_rejection_allows_one_repair_without_leaking_bad_claim(self):
        bad=claim('Состав не сохнет, потому что в нём недостаточно активатора.')
        generate=AsyncMock(side_effect=[draft([bad]),verdict([bad],ok=False),draft(),verdict()])
        audit={};out=await produce_answer(Q,evidence(),[],generate,audit=audit)
        self.assertEqual(out,PARAPHRASE)
        self.assertNotIn('активатор',out)
        self.assertTrue(audit['repaired'])
        self.assertEqual(generate.await_count,4)

    async def test_rejection_on_both_attempts_clarifies(self):
        generate=AsyncMock(side_effect=[draft(),verdict(ok=False),draft(),verdict(ok=False)])
        out=await produce_answer(Q,evidence(),[],generate)
        self.assertNotIn(PARAPHRASE,out)
        self.assertIn('Как сушишь',out)

    async def test_unknown_term_from_previous_error_is_corrected_not_defended(self):
        turns=[(Q,'Если используешь акриловый полимер, выбирай растворитель.')]
        generate=AsyncMock()
        for q in ['Что за акриловый полимер?','Никакого акрилового полимера у нас нет']:
            out=await produce_answer(q,evidence(),turns,generate)
            self.assertIn('моя ошибка',out)
            self.assertNotIn('однокомпонент',out)
        generate.assert_not_awaited()

    async def test_previous_generated_facts_absent_from_both_model_passes(self):
        turns=[(Q,'Секретная выдумка UnicornChemical')]
        generate=AsyncMock(side_effect=[draft(),verdict()])
        await produce_answer('А что ещё?',evidence(),turns,generate)
        for call in generate.await_args_list:
            self.assertNotIn('UnicornChemical',call.args[1])
            self.assertIn(Q,call.args[1])

    async def test_missing_evidence_followup_acknowledges_known_facts_without_reasking(self):
        turns=[(Q,'Как сушишь состав — естественно или лампой?')]
        generate=AsyncMock()
        out=await produce_answer('сушу лампой ик по 2 минуты',Evidence(()),turns,generate)
        self.assertIn('ИК',out);self.assertIn('один цикл: 2 мин',out)
        self.assertNotIn('Как сушишь',out)
        self.assertIn('Сколько таких циклов',out)
        generate.assert_not_awaited()

    async def test_json_ollama_payload(self):
        seen=[]
        def handler(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200,json={'message':{'content':'{}'}})
        original=httpx.AsyncClient
        with patch('local_models.httpx.AsyncClient',side_effect=lambda **kw: original(**kw,transport=httpx.MockTransport(handler))):
            await chat('http://127.0.0.1:11434','qwen3:4b-instruct',[],json_mode=True)
        self.assertEqual(seen[0]['format'],'json')
        self.assertEqual(seen[0]['options']['temperature'],0)
        self.assertFalse(seen[0]['think'])


class BotDialogueIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import tempfile,bot
        self.bot=bot
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        for name,value in [('ACCESS_DB',Path(self.tmp.name)/'access.sqlite'),('ADMINS',{999}),('PROVIDER','ollama'),('MODEL_GATE',asyncio.Semaphore(1))]:
            p=patch.object(bot,name,value);p.start();self.addCleanup(p.stop)
        bot.init_db();bot.grant(123,999,1)
        from types import SimpleNamespace
        self.message=SimpleNamespace(text=Q,reply_text=AsyncMock())
        self.update=SimpleNamespace(effective_user=SimpleNamespace(id=123),effective_chat=SimpleNamespace(id=123,type='private'),message=self.message)
        self.context=SimpleNamespace(user_data={},bot=SimpleNamespace(send_chat_action=AsyncMock()))
        self.hits=[{'id':1,'content':'№101: '+SOURCE+'\n№102: '+IR_SOURCE}]

    async def test_reported_two_turn_dialogue_through_telegram_handler(self):
        calls=[]
        async def generate(system,prompt):
            p=json.loads(prompt);calls.append(p)
            if system==VERIFY_SYSTEM: return verdict(p['draft']['claims'])
            if not p['case']['is_followup']: return draft(followup='drying_method')
            ir=next(u for u in p['available_units'] if 'ИК' in u['text'])
            c1=claim('Ты сушишь ИК-лампой по 2 минуты за цикл.',[],kind='observation',facts=['drying_method','cycle_duration'])
            c2=claim('При таком способе сушки толстый слой тоже может сохнуть дольше тонкого.',[ir['id']],kind='explanation',ident='c2')
            return draft([c1,c2],followup='cycle_count')
        retrieval=AsyncMock(return_value=self.hits)
        with patch.object(self.bot,'retrieve',retrieval),patch.object(self.bot,'generate_response',side_effect=generate):
            await self.bot.answer(self.update,self.context)
            first=self.message.reply_text.await_args.args[0]
            self.message.text='сушу лампой ик по 2 минуты'
            await self.bot.answer(self.update,self.context)
            second=self.message.reply_text.await_args.args[0]
        self.assertNotEqual(first,second)
        self.assertNotIn('Как сушишь',second)
        self.assertIn('по 2 минуты за цикл',second)
        self.assertIn('Сколько таких циклов',second)
        self.assertNotIn('«',second)
        self.assertNotEqual(retrieval.await_args_list[0],retrieval.await_args_list[1])
        self.assertIn('ИК',retrieval.await_args_list[1].kwargs['focus_query'])
        self.assertEqual(self.context.user_data['grounding']['status'],'grounded_synthesis')
        with self.bot.db() as con:
            self.assertEqual(con.execute('SELECT count FROM usage WHERE user_id=123').fetchone()[0],2)

    async def test_invalid_output_refunds_request(self):
        with patch.object(self.bot,'retrieve',AsyncMock(return_value=self.hits)),patch.object(self.bot,'generate_response',AsyncMock(return_value='Выдуманное объяснение.')):
            await self.bot.answer(self.update,self.context)
        with self.bot.db() as con:
            self.assertEqual(con.execute('SELECT count FROM usage WHERE user_id=123').fetchone()[0],0)
        self.assertNotIn('Выдуманное',self.message.reply_text.await_args.args[0])

    async def test_reset_clears_case(self):
        self.context.user_data.update(case={'facts':{'drying_method':{}}},turns=[('q','a')])
        await self.bot.new_topic(self.update,self.context)
        self.assertNotIn('case',self.context.user_data)
        self.assertNotIn('turns',self.context.user_data)

    async def test_evidence_not_disclosed_in_group(self):
        self.update.effective_user.id=999;self.update.effective_chat.type='group'
        self.context.user_data['grounding']={'selected':[{'text':'private quote','message_id':'1','passage_id':1}]}
        await self.bot.evidence_command(self.update,self.context)
        self.message.reply_text.assert_not_awaited()
