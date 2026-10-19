"""Loopback-only Ollama client shared by generation, indexing and evaluation."""
from __future__ import annotations
from urllib.parse import urlsplit
import httpx


def ollama_url(base: str, path: str) -> str:
    parsed = urlsplit(base)
    if (parsed.scheme not in ('http', 'https') or parsed.hostname not in ('localhost', '127.0.0.1', '::1')
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ('', '/')):
        raise ValueError('OLLAMA_BASE_URL должен указывать на локальную Ollama, например http://127.0.0.1:11434')
    return base.rstrip('/') + path


async def chat(base: str, model: str, messages: list[dict], *, legacy: bool = False,
               num_ctx: int = 8192, num_predict: int = 650, json_mode: bool = False) -> str:
    payload = {'model': model, 'messages': messages, 'stream': False, 'keep_alive': '10m',
               'options': {'num_ctx': num_ctx, 'num_predict': 850 if legacy else num_predict,
                           'seed': 42}}
    if not legacy:
        payload['options']['temperature'] = 0 if json_mode else .2
    if json_mode:
        payload['format'] = 'json'
    if not legacy and model.lower().startswith('qwen3'):
        payload['think'] = False
    async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=5), trust_env=False,
                                 follow_redirects=False) as client:
        response = await client.post(ollama_url(base, '/api/chat'), json=payload)
        if response.status_code == 400 and 'think' in payload:
            # Older/non-thinking Qwen tags can reject the optional think field.
            # Retry only this documented compatibility failure.
            error = response.json().get('error', '').lower()
            if 'think' in error and ('support' in error or 'unknown' in error):
                payload.pop('think')
                response = await client.post(ollama_url(base, '/api/chat'), json=payload)
        response.raise_for_status()
        result = response.json()
    content = result.get('message', {}).get('content', '')
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError('Модель вернула пустой ответ')
    return content.strip()


def embed(base: str, model: str, texts: list[str]) -> list[list[float]]:
    # CPU embeddings preserve the 3060 Ti's VRAM for the answering model.
    with httpx.Client(timeout=httpx.Timeout(300, connect=5), trust_env=False,
                      follow_redirects=False) as client:
        response = client.post(ollama_url(base, '/api/embed'), json={
            'model': model, 'input': texts, 'truncate': False, 'keep_alive': 0,
            'options': {'num_gpu': 0},
        })
        response.raise_for_status()
        vectors = response.json().get('embeddings', [])
    if len(vectors) != len(texts):
        raise ValueError('Ollama вернула неверное количество векторов')
    return vectors
