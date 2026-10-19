"""Local grounded synthesis: compose, check meaning, repair once, then clarify."""
from __future__ import annotations
import json
from conversation import build_case, relevant_history
from grounded_answers import Evidence, from_plain_text, correction_response
from answer_composer import (COMPOSE_SYSTEM, VERIFY_SYSTEM, available_questions,
                             public_case, parse_draft, parse_verdict, render_draft,
                             repeated_answer, clarification_answer)


async def produce_answer(question: str, evidence: Evidence | str, turns: list, generate,
                         audit: dict | None = None, case: dict | None = None,
                         previous_audit: dict | None = None) -> str:
    trace=audit if audit is not None else {}
    trace.clear()
    bundle=evidence if isinstance(evidence,Evidence) else from_plain_text(question,evidence)
    case=case if case is not None else build_case(question,turns)
    correction=correction_response(question,turns,bundle)
    if correction:
        trace['status']='unsupported_term'
        return correction
    if not bundle.units:
        trace['status']='no_complete_evidence'
        return clarification_answer(case)
    previous_sources=[]
    if case.get('is_followup') and previous_audit:
        old={(s.get('message_id'),s.get('text')) for s in previous_audit.get('selected',[])}
        previous_sources=[u.id for u in bundle.units if (u.message_id,u.text) in old]
    payload={'case':public_case(case),'previous_user_questions':[q[:500] for q,_ in relevant_history(question,turns)[-2:]],
             'available_units':bundle.model_data(),'allowed_questions':available_questions(case),
             'already_suggested_source_ids':previous_sources}
    errors=[]
    for attempt in range(2):
        trace['attempts']=attempt+1
        raw=await generate(COMPOSE_SYSTEM,json.dumps(payload,ensure_ascii=False))
        draft,issue=parse_draft(raw,bundle,case)
        if draft is not None and draft['status']!='answer':
            trace['status']=draft['status']
            return clarification_answer(case,conflicting=draft['status']=='conflicting')
        if draft is not None:
            answer=render_draft(draft,case)
            if repeated_answer(answer,turns,case):
                issue='answer_repeats_previous_turn'
            else:
                review_payload={'case':public_case(case),'available_units':bundle.model_data(),
                                'draft':draft,'already_suggested_source_ids':previous_sources,
                                'followup_text':available_questions(case).get(draft['followup'])}
                verdict=parse_verdict(await generate(VERIFY_SYSTEM,json.dumps(review_payload,ensure_ascii=False)),draft)
                if verdict and verdict['dialogue_ok'] and all(c['supported'] for c in verdict['claims']):
                    trace['status']='grounded_synthesis'
                    trace['claims']=draft['claims']
                    trace['followup']=draft['followup']
                    selected={i for c in draft['claims'] for i in c['source_ids']}
                    trace['selected']=[{'unit_id':u.id,'message_id':u.message_id,'passage_id':u.passage_id,
                                        'text':u.text,'context':u.context,'role':'support'}
                                       for u in bundle.units if u.id in selected]
                    trace['repaired']=bool(attempt)
                    return answer
                issue='invalid_review' if verdict is None else 'review_rejected: '+json.dumps(verdict,ensure_ascii=False)
        errors.append(issue)
        payload['repair']={'rejection':issue,'instruction':'Составь новый ответ с теми же источниками; устрани причину отклонения. При нехватке сведений insufficient.'}
    trace.update(status='unconfirmed',rejections=errors)
    return clarification_answer(case)
