"""Source-bound paraphrases and case-aware questions.

Python checks provenance, quantities and dialogue constraints. The local model
checks meaning; its review is useful but is not a proof of technical correctness.
"""
from __future__ import annotations
from difflib import SequenceMatcher
import re
from conversation import FACT_LABELS, next_clarification
from grounded_answers import (Evidence, parse_object, PERSONAL, PROFANITY,
                              SCHEME, phrase_supported, normalized)
from response_quality import META, quantities

from answer_prompts import COMPOSE_SYSTEM, VERIFY_SYSTEM

KINDS = frozenset(('explanation','possible_reason','check','observation'))


def available_questions(case: dict) -> dict[str,str]:
    result={}
    trial=dict(case)
    trial['asked_slots']=list(case.get('asked_slots',[]))
    for _ in range(len(FACT_LABELS)):
        item=next_clarification(trial)
        if not item: break
        slot,text=item
        result[slot]=text
        trial['asked_slots'].append(slot)
    return result


def public_case(case: dict) -> dict:
    return {k:case.get(k) for k in ('topic','latest_message','is_followup','facts','new_facts','changed_facts')}


def _list_of_ids(value, allowed, maximum):
    return (isinstance(value,list) and len(value)<=maximum
            and all(isinstance(v,str) and v in allowed for v in value)
            and len(set(value))==len(value))


def parse_draft(raw: str, evidence: Evidence, case: dict):
    value=parse_object(raw)
    if not value or set(value)!={'status','claims','followup'}:
        return None,'invalid_draft_schema'
    status,claims,followup=value['status'],value['claims'],value['followup']
    questions=available_questions(case)
    if (status not in ('answer','insufficient','conflicting') or not isinstance(claims,list)
            or len(claims)>4 or (followup is not None and (not isinstance(followup,str) or followup not in questions))):
        return None,'invalid_status_or_question'
    if (status=='answer')!=bool(claims):
        return None,'empty_or_unexpected_claims'
    units={u.id:u for u in evidence.units}
    facts=case.get('facts',{})
    ids=set()
    total=0
    for claim in claims:
        if not isinstance(claim,dict) or set(claim)!={'id','kind','text','source_ids','fact_keys'}:
            return None,'invalid_claim_schema'
        ident,kind,text=claim['id'],claim['kind'],claim['text']
        if not isinstance(ident,str) or not re.fullmatch(r'c[1-4]',ident) or ident in ids:
            return None,'invalid_claim_id'
        ids.add(ident)
        if not isinstance(kind,str) or kind not in KINDS or not isinstance(text,str) or not 15<=len(text)<=650:
            return None,'invalid_claim_text'
        total+=len(text)
        if total>1500 or '?' in text or META.search(text) or PERSONAL.search(text) or PROFANITY.search(text):
            return None,'non_student_text'
        if not _list_of_ids(claim['source_ids'],units,5) or not _list_of_ids(claim['fact_keys'],facts,8):
            return None,'unknown_evidence_or_fact'
        if kind=='observation':
            if not claim['fact_keys']: return None,'observation_without_user_fact'
        elif not claim['source_ids']:
            return None,'technical_claim_without_source'
        source=' '.join(units[i].text for i in claim['source_ids'])
        user=' '.join(facts[k]['evidence'] for k in claim['fact_keys'])
        if kind!='observation' and len(text)>65:
            plain=lambda s: normalized(re.sub(r'\W+',' ',s)).strip()
            if any(SequenceMatcher(None,plain(text),plain(units[i].text)).ratio()>.94
                   for i in claim['source_ids']):
                return None,'copied_source_wording_rephrase_without_changing_meaning'
        if quantities(text)-quantities(source+' '+user):
            return None,'unsupported_quantity'
        if SCHEME.search(text):
            return None,'unapproved_component_recipe'
        if re.search(r'(?i)\b(?:суши|грей|нагревай|прогревай|держи|выдержи|сделай|повтори)(?:те)?\b[^.!?]{0,70}\d',text):
            return None,'unapproved_numeric_regime'
        for term in re.findall(r'(?i)\b(?:акрил\w*|эпоксид\w*|полиуретан\w*|алкид\w*|однокомпонент\w*|двухкомпонент\w*)\s+[а-яё-]+',text):
            scoped=Evidence(tuple(units[i] for i in claim['source_ids']))
            if not phrase_supported(term,scoped):
                return None,'unsupported_material_name'
    if claims and all(c['kind']=='observation' for c in claims):
        return None,'no_technical_content'
    if case.get('is_followup') and case.get('new_facts') and claims:
        used={k for c in claims for k in c['fact_keys']}
        if not set(case['new_facts'])<=used:
            return None,'latest_facts_ignored'
        combined=' '.join(c['text'] for c in claims)
        for key,fact in case['new_facts'].items():
            if key=='drying_method':
                pattern={'infrared':r'\bик\b|инфракрас','lamp':r'ламп',
                         'natural':r'естествен|без\s+(?:нагрев|ламп)'}[fact['value']]
                if not re.search(pattern,combined,re.I): return None,'latest_method_not_addressed'
            if key=='application_method' and not re.search(r'кист' if fact['value']=='brush' else r'а[эе]рограф',combined,re.I):
                return None,'latest_application_not_addressed'
            if fact.get('unit') and (fact['value'],fact['unit']) not in quantities(combined):
                return None,'latest_measurement_not_addressed'
            if key=='cycle_duration' and not re.search(r'(?i)\bпо\s+\d|цикл|подход',combined):
                return None,'per_cycle_scope_lost'
            if key=='cycle_count' and not re.search(r'\b'+re.escape(fact['value'])+r'\s*(?:цикл|подход|раз)',combined,re.I):
                return None,'latest_cycle_count_not_addressed'
    return value,None


def parse_verdict(raw: str, draft: dict):
    value=parse_object(raw)
    if not value or set(value)!={'claims','dialogue_ok','reason'}:
        return None
    if type(value['dialogue_ok']) is not bool or not isinstance(value['reason'],str) or len(value['reason'])>600:
        return None
    checks=value['claims']
    expected={c['id'] for c in draft['claims']}
    if not isinstance(checks,list) or len(checks)!=len(expected): return None
    seen=set()
    for check in checks:
        if not isinstance(check,dict) or set(check)!={'id','supported','reason'}: return None
        ident=check['id']
        if not isinstance(ident,str) or ident not in expected or ident in seen: return None
        if type(check['supported']) is not bool or not isinstance(check['reason'],str) or len(check['reason'])>600: return None
        seen.add(ident)
    return value


def render_draft(draft: dict, case: dict) -> str:
    paragraphs=[c['text'].strip() for c in draft['claims']]
    if draft['followup']:
        paragraphs.append(available_questions(case)[draft['followup']])
    return '\n\n'.join(paragraphs)


def repeated_answer(answer: str, turns: list, case: dict) -> bool:
    if not turns or not case.get('is_followup'): return False
    old=normalized(re.sub(r'\W+',' ',turns[-1][1])).strip()
    new=normalized(re.sub(r'\W+',' ',answer)).strip()
    return bool(old and SequenceMatcher(None,old,new).ratio()>.84)


def clarification_answer(case: dict, *, conflicting: bool = False) -> str:
    facts=case.get('new_facts',{}) if case.get('is_followup') else {}
    labels=[]
    for key,fact in facts.items():
        if key=='drying_method': labels.append('сушка: '+fact['label'])
        elif key=='cycle_duration': labels.append('один цикл: '+fact['label'])
        elif key in ('cycle_count','total_elapsed','material_state','material_name'):
            labels.append(FACT_LABELS[key]+': '+fact['label'])
    lead=('Учёл: '+', '.join(labels)+'. ') if labels else ''
    text=('Есть противоречивые сведения; пока не могу выбрать подходящий способ действий.'
          if conflicting else 'Пока не хватает подтверждённых сведений, чтобы объяснить причину в твоём случае.')
    item=next_clarification(case)
    return lead+text+('\n\n'+item[1] if item else '')
