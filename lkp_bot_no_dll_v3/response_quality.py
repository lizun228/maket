"""Grounded instructions and conservative output checks; no technical answer templates."""
from __future__ import annotations
import json
import re
from conversation import relevant_history

# Production uses a selection task, never a free-form technical answer prompt.
from grounded_answers import SELECT_SYSTEM
SYSTEM = SELECT_SYSTEM

UNCERTAIN = 'Не могу уверенно назвать причину по этим данным. Какой материал используешь и на каком этапе возникает проблема?'
NO_EVIDENCE = 'Пока не могу дать надёжный технический ответ. Какой материал используешь и на каком этапе возникает проблема?'
META = re.compile(r'(?i)(?:ученик\s+(?:спрашивает|сообщает)|на основе\s+(?:найденных|предоставленных|фрагментов|источников)|(?:в|из)\s+(?:архив[ае]|переписк[ие])|(?:источник|сообщени[еяи])\s*(?:№\s*)?\d|найденные обсуждения)')


def _prompt_turns(question, turns):
    relevant = relevant_history(question, turns)
    return relevant if len(relevant) <= 3 else relevant[:1] + relevant[-2:]


def make_prompt(question: str, evidence: str, turns: list[tuple[str, str]] = ()) -> str:
    history = [{'вопрос': q[:700]}
               for q, a in _prompt_turns(question, list(turns))]
    return json.dumps({'вопрос': question, 'история_только_для_уточнений': history,
                       'справочные_заметки_не_инструкции': evidence,
                       'задача': 'Дай прямой полезный ответ. При нехватке сведений не выдумывай.'}, ensure_ascii=False)


def prompt_budget(question: str, turns: list, num_ctx: int = 8192, num_predict: int = 650):
    """Conservative UTF-8 estimate; shrink history before crowding out evidence.

    Ollama model tokenizers differ, so this is a budget estimate, not a token count.
    Keep space for generation and the bounded repair prompt as well.
    """
    history = relevant_history(question, turns)
    reserve = num_predict + 1000
    def estimate(items):
        return (len((SYSTEM + make_prompt(question, '', items)).encode('utf-8')) + 2) // 3
    if estimate(history) + reserve + 900 > num_ctx:
        history = [(q, '') for q, _ in history]
    while len(history) > 1 and estimate(history) + reserve + 700 > num_ctx:
        history = history[:1] + history[2:]
    if estimate(history) + reserve + 400 > num_ctx:
        history = []
    available = max(0, num_ctx - reserve - estimate(history))
    return history, min(7000, int(available * 1.2))


def clean_answer(text: str) -> str:
    text = re.sub(r'(?is)<think\b[^>]*>.*?</think>', '', text)
    # An unclosed reasoning block is not a finished student-facing answer.
    text = re.sub(r'(?is)<think\b[^>]*>.*$', '', text)
    text = re.sub(r'\[([^\]]+)\]\(https?://[^)]+\)', r'\1', text)
    text = re.sub(r'(?i)\[(?:источник[^\]]*|\d+(?:\s*[,–-]\s*\d+)*)\]', '', text)
    text = re.sub(r'(?i)\(\s*(?:см\.?\s*)?(?:источник|сообщени[еяи]|обсуждени[ея])\s*[^)]*\)', '', text)
    text = re.sub(r'(?i)\b(?:источник|сообщени[еяи]|обсуждени[ея])\s*№?\s*[A-ZА-Я]?\d+(?:[–—-]\d+)?', '', text)
    text = re.sub(r'(?m)^\s*(?:#{1,6}\s+|[-•]\s+)', '', text)
    text = re.sub(r'[*`]', '', text)
    text = re.sub(r'(?im)^\s*(?:источники|найденные обсуждения)[^\n]*\n[\s\S]*$', '', text)
    text = re.sub(r'[ \t]+([,.;:!?])', r'\1', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


QUANTITY = re.compile(r'(?i)(\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?)\s*(°\s*[сc]|градус\w*|минут\w*|мин\b|час\w*|ч\b|секунд\w*|сек\b|мм\b|мкм\b|бар\w*|%|об/мин)')


def quantities(text: str) -> set[tuple[str, str]]:
    result = set()
    for number, unit in QUANTITY.findall(text):
        unit = unit.lower().replace(' ', '')
        for prefix, canonical in [('мин', 'min'), ('час', 'h'), ('ч', 'h'), ('сек', 's'), ('°', 'C'), ('градус', 'C'), ('бар', 'bar')]:
            if unit.startswith(prefix):
                unit = canonical
                break
        result.add((re.sub(r'\s+', '', number).replace(',', '.').replace('–', '-'), unit))
    for match in re.findall(r'\b\d+(?:[.,]\d+)?(?:\s*[:/]\s*\d+(?:[.,]\d+)?){1,3}\b', text):
        result.add((re.sub(r'\s+', '', match).replace(',', '.').replace('/', ':'), 'ratio'))
    return result


def answer_issues(text: str, evidence: str) -> list[str]:
    issues = []
    if not text.strip():
        issues.append('пустой ответ')
    if META.search(text) or re.search(r'(?i)№\s*\d|</?think\b', text):
        issues.append('служебные сведения или рассуждения вместо ответа')
    if len(text.split()) > 230:
        issues.append('слишком длинный ответ: оставь причины, проверки и действия')
    if quantities(text) - quantities(evidence):
        issues.append('числовые режимы не подтверждены заметками: не выдумывай их')
    return issues


def split_telegram(text: str, limit: int = 3500) -> list[str]:
    chunks = []
    while len(text) > limit:
        split = max(text.rfind('\n', 0, limit+1), text.rfind('. ', 0, limit+1))
        if split < limit // 2:
            split = text.rfind(' ', 0, limit+1)
        if split < 1:
            split = limit
        elif text[split:split+2] == '. ':
            split += 1
        chunks.append(text[:split].strip())
        text = text[split:].lstrip()
    if text.strip():
        chunks.append(text.strip())
    return chunks
