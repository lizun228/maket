"""Immutable evidence units with local provenance and applicability filters."""
from __future__ import annotations
from dataclasses import dataclass
import json
import re
from search_engine import message_parts, normalized, query_terms, _terms_in


ANSWER_VERSION = 'dialogue-synthesis-v3'


@dataclass(frozen=True)
class Unit:
    id: str
    text: str
    context: str
    passage_id: int | str
    message_id: str

    def model_data(self):
        return {'id': self.id, 'text': self.text, 'context': self.context}


@dataclass(frozen=True)
class Evidence:
    units: tuple[Unit, ...]
    topic: str = ''

    def model_data(self):
        return [unit.model_data() for unit in self.units]


# Drop whole unsuitable sentences; never remove a negation/condition or replace
# words inside a technical claim to make it pass a guard.
PERSONAL = re.compile(r'https?://|@[a-z0-9_]{4,}|\+\d[\d ()-]{8,}', re.I)
PROFANITY = re.compile(r'(?i)\b(?:бля\w*|хер\w*|хуй\w*|хуе\w*|пизд\w*|заеб\w*|ебан\w*)')
VAGUE = re.compile(r'(?i)^(?:привет\w*|здравств\w*|спасибо|благодар\w*|да\b|нет\b|это\b|он\b|она\b|оно\b|так\b|тогда\b|первый\b|второй\b|они\b|тоже\b|понятно\b|там вариантов|соглас\w*)')
SCHEME = re.compile(r'(?i)(?:добав\w*|подмеш\w*|замен\w*|сократ\w*|увелич\w*|уменьш\w*|меша\w*|бодяж\w*|меняй|разбавля\w*)[^.!?]{0,100}(?:активатор|отвердител|разбавител|растворител|полимер|баз[ауые])|(?:активатор|отвердител|разбавител|растворител|полимер|баз[ауые])[^.!?]{0,70}(?:добав\w*|замен\w*|сократ\w*|меняй)')
AERO = re.compile(r'(?i)а[эе]рограф|\bаэро\b|дюз|распыл')
BRUSH = re.compile(r'(?i)кист')
DRY = re.compile(r'(?i)сох|суш|высых|высох|просох|резинов|мягк')
DRY_CAUSE = re.compile(r'(?i)температур|толщ|сло[йяе]|растворител|разбавител|баз[ауы]|сох|суш|высых|мягк|резинов|испар|выдерж')


def problem_kind(question: str) -> str:
    q = normalized(question)
    if DRY.search(q) and re.search(r'долг|дольш|медлен|\bне\b|мягк|резинов', q):
        return 'drying'
    return ''


def policy_issue(question: str, text: str, context: str) -> str | None:
    if '?' in text or text.rstrip().endswith((':', ',', ';')):
        return 'question_or_incomplete'
    if len(text) < 35 or not re.search(r'[а-яa-z]', text, re.I):
        return 'too_short'
    if VAGUE.search(text) or re.search(r'(?i)^а\s+с\s+|на последнем месте',text) or PERSONAL.search(text) or PROFANITY.search(text):
        return 'not_standalone_or_private'
    if '[Вложение' in text or re.search(r'(?i)источник|сообщени[еяи]\s*№|№\s*\d', text):
        return 'not_textual_evidence'
    if re.search(r'(?i)\b(?:суши|прогрей|нагревай|выдержи|держи)(?:те)?\b[^.!?]{0,70}(?:\d|на ночь|сутк)',text):
        return 'unapproved_product_regime'
    # Current archive does not include verified product-specific technical maps.
    # Do not publish recipes/substitutions just because a participant tried one.
    if SCHEME.search(text):
        return 'unapproved_recipe'
    if BRUSH.search(question) and not AERO.search(question) and (AERO.search(text) or (AERO.search(context) and not BRUSH.search(text))):
        return 'different_application_method'
    for chemistry in ('акрил', 'эпоксид', 'полиуретан', 'алкид', 'однокомпонент', 'двухкомпонент'):
        if chemistry in normalized(text) and chemistry not in normalized(question):
            return 'unknown_material_chemistry'
    if re.search(r'(?i)пластик',text) and not re.search(r'(?i)пластик',question):
        return 'unknown_substrate'
    if re.search(r'(?i)\b(?:она|он|оно|они)\b',text) and not re.search(r'(?i)состав|материал|полимер|баз[ауы]|краск|смесь',text):
        return 'ambiguous_subject'
    if re.search(r'(?i)х\.{1,3}н',text):
        return 'profanity'
    if problem_kind(question) == 'drying':
        if not DRY.search(context) or not DRY_CAUSE.search(text):
            return 'different_problem'
        # An isolated conditional claim isn't a diagnosis or a reason to add
        # thinner to the pupil's unknown mixture.
        if re.search(r'(?i)(?:без\s+разбав|(?:не\s+добав\w*)[^.!?]{0,30}разбав)', text):
            return 'unconfirmed_additive_cause'
    return None


def build_evidence(question: str, hits: list[dict], max_chars: int = 7000) -> Evidence:
    candidates, seen = [], set()
    terms = query_terms(question)
    for hit_rank,item in enumerate(hits):
        messages = message_parts(item['content'])
        for index, (mid, body) in enumerate(messages):
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', body) if s.strip()]
            previous = ' '.join(text for _, text in messages[max(0,index-2):index])[-420:]
            context = (previous+'\n'+body).strip()
            choices=[]
            for number,text in enumerate(sentences):
                tail = len(body)>=499 and number==len(sentences)-1 and not re.search(r'[.!]$',text)
                if not tail:
                    choices.append(text)
                if number+1<len(sentences):
                    next_tail = len(body)>=499 and number+1==len(sentences)-1 and not re.search(r'[.!]$',sentences[number+1])
                    if not tail and not next_tail:
                        choices.append(text+' '+sentences[number+1])
            for text in choices:
                if len(text)>650 or policy_issue(question,text,context):
                    continue
                if not _terms_in(context,terms):
                    continue
                key=normalized(text)
                if key in seen:
                    continue
                seen.add(key)
                matches=_terms_in(text,terms)
                score=3*len(matches)+max(0,6-hit_rank)*.3
                if problem_kind(question)=='drying':
                    score+=2*bool(re.search(r'(?i)температур|толщ|проверь|попроб|испар|сох',text))
                # Prefer explicit full checks over an isolated follow-up pronoun.
                score+=.4*bool(re.search(r'(?i)проверь|попроб',text))
                candidates.append((score,text,context,item.get('id',''),mid))
    candidates.sort(key=lambda x:-x[0])
    units=[]
    for _,text,context,pid,mid in candidates:
        # End of own message includes a complete source, rather than a clipped
        # prefix whose missing condition might change the advice.
        context_sentences=[s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+',context) if s.strip()]
        model_context=' '.join(s for s in context_sentences if len(s)<=900)
        if len(model_context)>1000:
            model_context=text
            for sentence in reversed(context_sentences):
                if sentence not in model_context and len(sentence)+len(model_context)+1<=1000:
                    model_context=sentence+' '+model_context
        unit=Unit(f'u{len(units)+1}',text,model_context,pid,mid)
        trial=units+[unit]
        if len(json.dumps([u.model_data() for u in trial],ensure_ascii=False))>max_chars:
            continue
        units.append(unit)
        if len(units)>=24:
            break
    return Evidence(tuple(units),question)


def from_plain_text(question: str, text: str, max_chars: int = 7000) -> Evidence:
    """Compatibility for offline callers; production uses original message IDs."""
    return build_evidence(question,[{'content':text,'id':'local'}],max_chars)


def parse_object(raw: str) -> dict | None:
    # No extraction of arbitrary braces from prose: valid JSON is required.
    if not isinstance(raw,str) or len(raw) > 12000:
        return None
    try:
        value=json.loads(raw)
    except (ValueError,TypeError):
        return None
    return value if isinstance(value,dict) else None


def named_term(question: str) -> str | None:
    pattern=r'(?i)(?:что\s+за|какой\s+такой|что\s+такое|никакого|никакой|не\s+использую|ты\s+назвал(?:\s+состав)?)\s+([а-яёa-z][а-яёa-z-]*(?:\s+[а-яёa-z-]+)?)'
    match=re.search(pattern,question)
    return match.group(1).strip().rstrip(' .?!') if match else None


def _phrase_pattern(term: str) -> str:
    words=re.findall(r'[а-яёa-z]+',normalized(term))
    return r'\b'+r'\w*\s+'.join(re.escape(w[:6]) for w in words)+r'\w*\b'


def phrase_supported(term: str, evidence: Evidence) -> bool:
    # Attribute and noun must be adjacent in ONE actual sentence, not different
    # words pooled from thinner, paint and polymer descriptions.
    if len(re.findall(r'[а-яёa-z]+',term))<2:
        return True
    pattern=_phrase_pattern(term)
    return any(re.search(pattern,normalized(unit.text)) for unit in evidence.units)


def correction_response(question: str, turns: list, evidence: Evidence) -> str | None:
    term=named_term(question)
    if not term:
        return None
    pattern=_phrase_pattern(term)
    introduced=bool(turns and re.search(pattern,normalized(turns[-1][1]))
                    and not any(re.search(pattern,normalized(q)) for q,_ in turns))
    denial=bool(re.search(r'(?i)никак|не\s+использую|не\s+применя|у\s+нас\s+нет',question))
    if not denial and phrase_supported(term,evidence):
        return None
    if introduced:
        return 'Я использовал неподтверждённое название состава. Это моя ошибка; прежнее объяснение не следует принимать за описание твоего материала. Как состав называется в обучении или на упаковке?'
    if denial:
        return 'Понял: не будем приписывать твоему составу этот тип материала. Как он называется в обучении или на упаковке?'
    return 'Не могу подтвердить такое название материала для твоей работы. Как он называется в обучении или на упаковке?'
