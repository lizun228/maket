"""Bounded dialogue context: user clarifications, topic changes and expiry."""
from __future__ import annotations
import re
import time
from search_engine import query_terms

HISTORY_TTL = 30 * 60
MAX_TURNS = 4


def relevant_history(question: str, turns: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if not turns:
        return []
    if re.match(r'(?i)^\s*(?:другой\s+вопрос|новая\s+тема|теперь\s+(?:про|о)\b)',question):
        return []
    terms = set(query_terms(question))
    old = set(query_terms(' '.join(q for q, _ in turns[-2:])))
    continuation = bool(re.match(r'(?i)^\s*(?:а\s+(?:если|когда|как|можно|без|с|сколько|что)|тогда|а это|а так|почему так|он\b|она\b|оно\b|это\b|тот\b|этот\b|сколько\s+(?:ждать|сушить)|и\s+(?:как|что)|то есть|да[, ]|нет[, ])', question))
    pending_answer = turns[-1][1].rstrip().endswith('?')
    explicit_question = bool(re.match(r'(?i)^\s*(?:как|почему|чем|зачем|можно|что|сколько|какой|какая)\b', question))
    clarification = pending_answer and len(question) < 180 and '?' not in question and not explicit_question
    # A full explicit question starts its own topic, even when it is short.
    factual_reply = (not explicit_question and len(question)<220 and
                     bool(re.match(r'(?i)^\s*(?:сушу|сушим|наношу|наносим|нан[её]с|сделал|после|при|остал\w*|температура|в боксе|прошло|уже|всего|состав|полимер|база|\d+)\b',question)))
    if clarification or continuation or factual_reply or (len(terms) <= 2 and terms & old):
        return turns[-MAX_TURNS:]
    return []


def search_question(question: str, turns: list[tuple[str, str]]) -> str:
    relevant = relevant_history(question, turns)
    if not relevant:
        return question
    # Only user questions can define retrieval. Previous model answers are not
    # evidence and must never feed generated technical claims back into search.
    context = list(dict.fromkeys([relevant[0][0]] + [q for q, _ in relevant[-2:]]))
    return ' '.join(context) + ' ' + question


def load_turns(data: dict, now: float | None = None) -> list[tuple[str, str]]:
    now = time.time() if now is None else now
    if now - data.get('turns_at', 0) > HISTORY_TTL:
        data.pop('turns', None)
        data.pop('case', None)
        return []
    return data.get('turns', [])[-MAX_TURNS:]


def remember(data: dict, question: str, answer: str, turns: list[tuple[str, str]]) -> None:
    if data.get('case',{}).get('latest_message') != question:
        data['case'] = build_case(question,turns,data.get('case'))
    updated = relevant_history(question, turns) + [(question, answer)]
    data['turns'] = updated if len(updated) <= MAX_TURNS else updated[:1] + updated[-(MAX_TURNS-1):]
    data['turns_at'] = time.time()


# Facts are observations supplied by the user, not technical knowledge. Numeric
# scope is explicit: a per-cycle duration must never become total elapsed time.
FACT_LABELS = {
    'application_method':'способ нанесения', 'drying_method':'способ сушки',
    'cycle_duration':'длительность одного цикла', 'heating_duration':'время нагрева (область не уточнена)',
    'total_elapsed':'общее прошедшее время', 'cycle_count':'число циклов',
    'temperature':'температура участка', 'ambient_temperature':'температура в помещении',
    'material_state':'состояние материала', 'material_name':'название состава',
}


def extract_facts(text: str) -> dict:
    facts={}
    t=text.lower().replace('ё','е')
    hypothetical=bool(re.match(r'^\s*(?:а\s+если|если|можно\s+ли|что\s+если|как\s+(?:сушить|наносить))',t))
    if hypothetical:
        return facts
    if re.search(r'кист',t) and not re.search(r'(?:не|без)\s+кист',t):
        facts['application_method']={'value':'brush','label':'кисть','evidence':text}
    if re.search(r'а[эе]рограф',t) and not re.search(r'(?:не|без)\s+а[эе]рограф',t):
        facts['application_method']={'value':'airbrush','label':'аэрограф','evidence':text}
    natural=bool(re.search(r'(?<!не )естествен|без\s+(?:ик|ламп|нагрев)',t))
    ir=bool(re.search(r'\bик\b|инфракрас',t)) and not bool(re.search(r'не\s+(?:ик|инфракрас)',t))
    lamp=bool(re.search(r'ламп',t))
    if natural:
        facts['drying_method']={'value':'natural','label':'естественная сушка / без нагрева','evidence':text}
    elif ir:
        facts['drying_method']={'value':'infrared','label':'ИК-лампа','evidence':text}
    elif lamp:
        facts['drying_method']={'value':'lamp','label':'лампа, тип не уточнён','evidence':text}
    durations=re.finditer(r'(?P<prefix>по\s+)?(?P<number>\d+(?:[.,]\d+)?)\s*(?P<unit>минут\w*|мин\b|секунд\w*|сек\b|час\w*|ч\b)',t)
    for duration in durations:
        number=duration['number'].replace(',','.')
        unit=duration['unit']
        canonical='min' if unit.startswith('мин') else 's' if unit.startswith('сек') else 'h'
        shown='мин' if canonical=='min' else 'сек' if canonical=='s' else 'ч'
        before=t[max(0,duration.start()-35):duration.start()]
        if duration['prefix']:
            slot='cycle_duration'
        elif re.search(r'всего|уже|прошло|общ\w*\s+врем|с\s+начала',before):
            slot='total_elapsed'
        elif re.search(r'цикл|подход|прогрев',before):
            slot='cycle_duration'
        else:
            slot='heating_duration'
        facts[slot]={'value':number,'unit':canonical,'label':number+' '+shown,'evidence':text}
    count=re.search(r'\b(\d+)\s*(?:цикл\w*|раз(?:а)?|подход\w*)\b',t)
    if count:
        facts['cycle_count']={'value':count[1],'label':count[1]+' циклов/подходов','evidence':text}
    temperature=re.search(r'(-?\d+(?:[.,]\d+)?)\s*(?:°\s*[сc]?|градус\w*)',t)
    if temperature:
        slot='ambient_temperature' if re.search(r'бокс|помещен|воздух|гараж',t) else 'temperature'
        facts[slot]={'value':temperature[1].replace(',','.'),'unit':'C','label':temperature[1]+' °C','evidence':text}
    states=list(dict.fromkeys(re.findall(r'(?:не\s+)?(?:лип(?:нет|кий|кость)|продавлива\w*|резинов\w*|мягк\w*|тв[её]рд\w*)',t)))
    if states:
        state='; '.join(states)
        facts['material_state']={'value':state,'label':state,'evidence':text}
    name=re.search(r'(?:состав|материал|полимер)\s+(?:называется|это|марки)\s+([^.!?]{2,80})',text,re.I)
    if name:
        facts['material_name']={'value':name[1].strip(),'label':name[1].strip(),'evidence':text}
    return facts


def asked_slots(answer: str) -> set[str]:
    question=answer[answer.rfind('\n')+1:] if '?' in answer else ''
    slots=set()
    if re.search(r'как\s+суш|естествен\w*\s+или|ламп\w*\s+или',question,re.I): slots.add('drying_method')
    if re.search(r'сколько.*цикл|сколько.*подход',question,re.I): slots.add('cycle_count')
    if re.search(r'сколько\s+времени.*прош|как\s+давно',question,re.I): slots.add('total_elapsed')
    if re.search(r'лип|продавлива|оста[её]тся\s+мяг',question,re.I): slots.add('material_state')
    if re.search(r'какой\s+(?:состав|материал)|как.*называется',question,re.I): slots.add('material_name')
    if re.search(r'температур',question,re.I):
        slots.add('ambient_temperature' if re.search(r'бокс|помещен|гараж',question,re.I) else 'temperature')
    return slots


def build_case(question: str, turns: list[tuple[str,str]], previous: dict | None = None) -> dict:
    related=relevant_history(question,turns)
    if related:
        facts=dict((previous or {}).get('facts',{}))
        topic=(previous or {}).get('topic') or related[0][0]
        for old_q,_ in related:
            # Persistent current facts win over earlier questions replayed here.
            for slot,fact in extract_facts(old_q).items(): facts.setdefault(slot,fact)
    else:
        facts={}
        topic=question
    latest=extract_facts(question)
    if (related and 'material_name' in asked_slots(related[-1][1]) and not latest
            and 2<=len(question)<=90 and '?' not in question
            and not re.search(r'(?i)не знаю|не помню|а если|никак|не использую',question)):
        latest['material_name']={'value':question.strip(),'label':question.strip(),'evidence':question}
    changed={k: {'old':facts[k],'new':v} for k,v in latest.items() if k in facts and
             (facts[k].get('value'),facts[k].get('unit'))!=(v.get('value'),v.get('unit'))}
    facts.update(latest)
    if 'drying_method' in changed:
        for slot in ('cycle_duration','heating_duration','cycle_count'):
            if slot not in latest: facts.pop(slot,None)
    # An explicit later method supersedes the old method. Never average or mix.
    pending=set((previous or {}).get('asked_slots',[])) if related else set()
    if related:
        for _,answer in related: pending.update(asked_slots(answer))
    state={'topic':topic,'latest_message':question,'facts':facts,'new_facts':latest,
           'changed_facts':changed,'asked_slots':sorted(pending),
           'is_followup':bool(related)}
    focus=[]
    if 'drying_method' in latest:
        focus.append('сушка ИК' if facts['drying_method']['value']=='infrared' else 'сушка')
    if any(k in latest for k in ('cycle_duration','total_elapsed','heating_duration','cycle_count')):
        if facts.get('drying_method',{}).get('value')=='infrared': focus.append('ИК сушка нагрев остывание')
        else: focus.append('сушка')
    if any(k in latest for k in ('temperature','ambient_temperature')): focus.append('сушка температура')
    if 'material_state' in latest: focus.append('сушка мягкий')
    state['focus_query']=' '.join(dict.fromkeys(focus))
    known=' '.join(('естественная сушка' if fact.get('value')=='natural' else fact['label'])
                   for key,fact in facts.items() if key in ('application_method','drying_method','material_state','material_name'))
    retrieval_topic=topic
    if 'application_method' in facts:
        retrieval_topic=re.sub(r'(?i)кист\w*|а[эе]рограф\w*','',retrieval_topic)
    if 'drying_method' in facts:
        retrieval_topic=re.sub(r'(?i)\bик\b|инфракрас\w*|ламп\w*|естествен\w*','',retrieval_topic)
    # Numbers remain structured observations; searching "2 minutes" must not
    # favour a coincidental regimen over the actual symptom and drying method.
    retrieval_reply=re.sub(r'\d+(?:[.,]\d+)?\s*(?:минут\w*|мин\b|секунд\w*|сек\b|час\w*|цикл\w*|раз\b)', '',question,flags=re.I)
    if 'application_method' in facts:
        retrieval_reply=re.sub(r'(?i)кист\w*|а[эе]рограф\w*','',retrieval_reply)
    if 'drying_method' in facts:
        retrieval_reply=re.sub(r'(?i)\bик\b|инфракрас\w*|ламп\w*|естествен\w*','',retrieval_reply)
    state['search_query']=' '.join(dict.fromkeys([retrieval_topic,retrieval_reply,known]))
    return state


def next_clarification(case: dict, avoid_previously_asked: bool = True) -> tuple[str,str] | None:
    facts=case.get('facts',{})
    excluded=set(facts)
    if avoid_previously_asked: excluded.update(case.get('asked_slots',[]))
    drying=bool(re.search(r'сох|суш|резинов|мягк',case.get('topic',''),re.I))
    options=[]
    if drying:
        options=[('drying_method','Как сушишь состав — естественно или лампой?'),
                 ('cycle_count','Сколько таких циклов нагрева уже сделал?'),
                 ('material_state','Материал остаётся липким или легко продавливается?'),
                 ('material_name','Как называется состав, который наносишь?'),
                 ('temperature','Измерял температуру участка во время нагрева?')]
        if facts.get('drying_method',{}).get('value') not in ('infrared','lamp'):
            options=[o for o in options if o[0] not in ('cycle_count','temperature')]
    elif re.search(r'каттер|катер|выдира|выдера|подрез',case.get('topic',''),re.I):
        options=[('material_state','Материал перед подрезкой твёрдый или остаётся мягким?'),
                 ('drying_method','Как сушишь состав — естественно или лампой?'),
                 ('material_name','Как называется состав, который подрезаешь?'),
                 ('application_method','Каким способом наносишь материал?')]
    else:
        options=[('material_name','Как называется материал, с которым работаешь?'),
                 ('application_method','Каким способом наносишь материал?')]
    return next((item for item in options if item[0] not in excluded),None)
