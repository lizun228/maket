import re

def clean_answer(text: str) -> str:
    """Format model output as plain Telegram text and remove citation artifacts."""
    text = re.sub(r'[*`#]', '', text)
    text = re.sub(r'(?i)\(\s*(?:см\.?\s*)?(?:источник|сообщени[ея]|обсуждени[ея]|№)\s*[^)]*\)', '', text)
    text = re.sub(r'(?i)\b(?:источник|сообщени[ея]|обсуждени[ея])\s*№?\s*[A-ZА-Я]?\d+(?:[–—-]\d+)?', '', text)
    text = re.sub(r'(?i)\b(?:см\.?\s*)?№\s*\d+(?:[–—-]\d+)?', '', text)
    text = re.split(r'(?im)^\s*(?:📚\s*)?найденные обсуждения(?: из архива)?\s*:', text, maxsplit=1)[0]
    text = re.sub(r'(?m)^[ \t]*[-•]\s+', '', text)
    text = re.sub(r'(?m)^[ \t]*[✅❗🔹📚]+\s*', '', text)
    text = re.sub(r'[ \t]+([,.;:!?])', r'\1', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
