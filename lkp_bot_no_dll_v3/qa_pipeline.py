"""Testable grounded answering pipeline, independent from Telegram and credentials."""
from __future__ import annotations
import json
from response_quality import SYSTEM, NO_EVIDENCE, UNCERTAIN, make_prompt, clean_answer, answer_issues, META


async def produce_answer(question: str, evidence: str, turns: list, generate) -> str:
    if not evidence.strip():
        return NO_EVIDENCE
    prompt = make_prompt(question, evidence, turns)
    raw = await generate(SYSTEM, prompt)
    answer = clean_answer(raw)
    issues = answer_issues(answer, evidence)
    if META.search(raw):
        issues.append('Убери ссылки на служебные сведения; переформулируй предложения естественно')
    if issues:
        # One bounded repair, with original evidence still present. Never add
        # model-written facts to the knowledge base or retrieve using the draft.
        repair = json.loads(prompt)
        repair['черновик_не_источник_фактов'] = answer[:1200]
        repair['исправления'] = issues
        repair['задача'] = 'Исправь черновик. Выведи только ответ ученику, без описания исправлений. Убери неподтверждённые параметры, не заменяй их другими догадками.'
        raw = await generate(SYSTEM, json.dumps(repair, ensure_ascii=False))
        answer = clean_answer(raw)
        if answer_issues(answer, evidence) or META.search(raw):
            return UNCERTAIN
    return answer or UNCERTAIN
