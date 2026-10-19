"""Select and check immutable source sentences; never forward model-written facts."""
from __future__ import annotations
import json
from conversation import relevant_history
from grounded_answers import (Evidence, SELECT_SYSTEM, REVIEW_SYSTEM, from_plain_text,
                              correction_response, parse_selection, parse_review,
                              policy_issue, render, clarification)
from response_quality import NO_EVIDENCE


def uncertain(question: str) -> str:
    return 'Пока нельзя уверенно выбрать причину или способ действий. '+clarification(question)


async def produce_answer(question: str, evidence: Evidence | str, turns: list, generate,
                         audit: dict | None = None) -> str:
    trace=audit if audit is not None else {}
    trace.clear()
    bundle=evidence if isinstance(evidence,Evidence) else from_plain_text(question,evidence)
    topic=bundle.topic or question
    correction=correction_response(question,turns,bundle)
    if correction:
        trace['status']='unsupported_term'
        return correction
    if not bundle.units:
        trace['status']='no_complete_evidence'
        return NO_EVIDENCE
    # Model replies are deliberately absent, including from the review pass.
    history=[q[:700] for q,_ in relevant_history(question,turns)[-3:]]
    payload={'question':question,'topic':topic,'previous_user_questions':history,
             'available_units':bundle.model_data()}
    raw=await generate(SELECT_SYSTEM,json.dumps(payload,ensure_ascii=False))
    selection=parse_selection(raw,bundle)
    if selection is None:
        trace['status']='invalid_selection'
        return uncertain(topic)
    status,selected=selection
    if status!='answer':
        trace['status']=status
        return uncertain(topic)
    # Reapply deterministic scope/recipe guards, independent of model confidence.
    if any(policy_issue(topic,unit.text,unit.context) for unit,_ in selected):
        trace['status']='policy_rejected'
        return uncertain(topic)
    review_payload={'question':question,'topic':topic,'previous_user_questions':history,
                    'selected':[dict(unit.model_data(),role=role) for unit,role in selected]}
    raw=await generate(REVIEW_SYSTEM,json.dumps(review_payload,ensure_ascii=False))
    reviewed=parse_review(raw,selected)
    if reviewed is None:
        trace['status']='invalid_review'
        return uncertain(topic)
    review_status,approved=reviewed
    if review_status!='ok':
        trace['status']='review_'+review_status
        return uncertain(topic)
    final=[(unit,role) for unit,role in selected if unit.id in approved]
    trace['status']='source_sentences'
    trace['selected']=[{'message_id':u.message_id,'passage_id':u.passage_id,
                        'text':u.text,'context':u.context,'role':role} for u,role in final]
    # Only immutable, program-constructed units and fixed nontechnical labels
    # enter the answer. No `answer`, `reason` or free prose from the model does.
    return render(topic,final)
