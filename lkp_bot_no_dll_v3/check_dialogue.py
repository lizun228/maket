"""Exercise complete conversations. --generate uses ONLY loopback Ollama.

The default checks retrieval/state without inventing model answers. Reports stay
in evaluation/results; they can contain private archive excerpts. Never upload.
"""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from dotenv import load_dotenv
import httpx
from conversation import build_case, remember
from conversation import asked_slots
from grounded_answers import build_evidence, ANSWER_VERSION
from local_models import chat, ollama_url
from qa_pipeline import produce_answer
from response_quality import prompt_budget
from search_engine import search_db

ROOT=Path(__file__).resolve().parent
DIALOGUES=[
    ('Пример владельца: сушка',[
        'состав который я наношу кисточкой долго сохнет',
        'сушу лампой ик по 2 минуты',
        'сделал 4 цикла, материал всё ещё мягкий']),
    ('Подрезка: проверка уточнения',[
        'Почему материал выдирается при подрезании каттером?',
        'после сушки материал остаётся мягким']),
]


async def run(database: Path,model: str,base: str,generate_answers: bool):
    if generate_answers:
        async with httpx.AsyncClient(timeout=10,trust_env=False,follow_redirects=False) as client:
            r=await client.get(ollama_url(base,'/api/tags'));r.raise_for_status()
            names={m['name'] for m in r.json().get('models',[])}
            if model not in names and model+':latest' not in names:
                raise ValueError('Модель не установлена')
    report={'version':ANSWER_VERSION,'model':model,'generation':'completed' if generate_answers else 'not_run',
            'note':'Автоматические признаки не заменяют оценку преподавателя. До: повтор на примере владельца; после: реальные ответы только при --generate.',
            'observed_before':{'source':'user_report','repeated_after_ir_clarification':True,'asked_known_drying_method':True},
            'dialogues':[]}
    for name,questions in DIALOGUES:
        data={};previous_audit=None;rows=[]
        for q in questions:
            tick=time.monotonic();turns=data.get('turns',[])
            case=build_case(q,turns,data.get('case'));data['case']=case
            hits=search_db(database,case['search_query'],6,focus_query=case['focus_query'])
            _,budget=prompt_budget(q,turns)
            bundle=build_evidence(case['search_query'],hits,max_chars=budget)
            audit={};answer=None
            if generate_answers:
                async def generate(system,prompt):
                    return await chat(base,model,[{'role':'system','content':system},{'role':'user','content':prompt}],
                                      num_predict=max(1000,int(os.environ.get('OLLAMA_NUM_PREDICT','1200'))),json_mode=True)
                answer=await produce_answer(q,bundle,turns,generate,audit=audit,case=case,previous_audit=previous_audit)
                checks={'different_from_previous':not turns or answer!=turns[-1][1],
                        'no_question_about_known_fact':not bool(asked_slots(answer)&set(case['facts'])),
                        'no_invented_acrylic_polymer':not re.search(r'(?i)акрил\w*\s+полимер',answer),
                        'grounded_synthesis':audit.get('status')=='grounded_synthesis'}
                remember(data,q,answer,turns);previous_audit=audit
                print('\n'+q+'\n'+answer,flush=True)
            else:
                checks={'has_evidence':bool(bundle.units),'keeps_topic':case['topic']==questions[0]}
                # Retain user-only history for subsequent state/retrieval checks.
                remember(data,q,'',turns)
                print(name,': фрагментов',len(hits),'пригодных реплик',len(bundle.units),flush=True)
            rows.append({'question':q,'case':case,'passage_ids':[h['id'] for h in hits],
                         'answer':answer,'checks':checks,'audit':audit,'elapsed_seconds':round(time.monotonic()-tick,2)})
        report['dialogues'].append({'name':name,'turns':rows})
    return report


def main():
    load_dotenv(ROOT/'.env')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generate',action='store_true')
    p.add_argument('--database',type=Path,default=ROOT/'knowledge.sqlite')
    p.add_argument('--model',default=os.environ.get('AI_MODEL','qwen3:4b-instruct'))
    p.add_argument('--base-url',default=os.environ.get('OLLAMA_BASE_URL','http://127.0.0.1:11434'))
    args=p.parse_args()
    try:
        report=asyncio.run(run(args.database,args.model,args.base_url,args.generate))
    except Exception as exc:
        print('Проверка не завершена:',type(exc).__name__)
        print('Для генерации запусти локальную Ollama; имя модели должно совпадать с ollama list. Архив никуда не отправлен.')
        return 1
    folder=ROOT/'evaluation'/'results';folder.mkdir(parents=True,exist_ok=True)
    stem='dialogue_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    with (folder/(stem+'.json')).open('x',encoding='utf-8') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    with (folder/(stem+'.txt')).open('x',encoding='utf-8') as f:
        f.write('Версия: '+ANSWER_VERSION+'\n'+report['note']+'\n')
        for dialogue in report['dialogues']:
            f.write('\n'+dialogue['name']+'\n')
            for row in dialogue['turns']:
                f.write('\nВопрос: '+row['question']+'\nОтвет: '+(row['answer'] or 'Генерация не запускалась')+'\n')
                f.write('Автопроверки: '+json.dumps(row['checks'],ensure_ascii=False)+'\n')
    print('\nОтчёт для чтения:',folder/(stem+'.txt'))
    print('Технический отчёт с основаниями:',folder/(stem+'.json'))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
