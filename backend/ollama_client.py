"""Local Ollama transport. No external-provider fallback."""
import json
import os
from pathlib import Path

import httpx


def settings():
    config = json.loads(Path(__file__).with_name('llm.json').read_text(encoding='utf-8'))
    return {
        'provider': os.getenv('ULCA_LLM_PROVIDER', config['provider']),
        'base_url': os.getenv('OLLAMA_BASE_URL', config['base_url']).rstrip('/'),
        'model': os.getenv('OLLAMA_MODEL', config['model']),
        'timeout_seconds': config['timeout_seconds'],
    }


async def complete_ollama(instructions, content, schema=None):
    config = settings()
    message = {'role': 'user', 'content': '\n\n'.join(
        part['text'] for part in content if part['type'] == 'input_text')}
    images = []
    for part in content:
        if part['type'] == 'input_image':
            url = part['image_url']
            if not url.startswith('data:image/') or ';base64,' not in url:
                raise ValueError('Expected base64 image')
            images.append(url.split(';base64,', 1)[1])
    if images:
        message['images'] = images
    if schema:
        instructions += '\nJSONのみを返してください。Markdownのコードフェンスや説明文は不要です。'
    body = {'model': config['model'], 'stream': False, 'think': False,
            'messages': [{'role': 'system', 'content': instructions}, message],
            'options': {'temperature': 0, 'num_predict': 2048}}
    if schema:
        body['format'] = schema
    async with httpx.AsyncClient(timeout=httpx.Timeout(config['timeout_seconds'], connect=10),
                                 trust_env=False) as client:
        response = await client.post(config['base_url']+'/api/chat', json=body)
        response.raise_for_status()
        payload = response.json()
    if payload.get('error'):
        raise ValueError('Ollama returned an error')
    if payload.get('done_reason') == 'length':
        raise ValueError('Ollama response was truncated')
    text = payload.get('message', {}).get('content', '')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Empty Ollama response')
    text = text.strip()
    # Some compatible MLX servers wrap even schema responses in Markdown.
    if schema and text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    return text
