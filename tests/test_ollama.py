import asyncio, json
import httpx
import pytest
from backend import ollama_client as ollama

@pytest.mark.parametrize('image', [False, True])
def test_ollama_request_and_json_fences(monkeypatch,image):
    captured=[]
    def handler(req):
        captured.append(req)
        return httpx.Response(200,json={'message':{'content':'```json\n{"ok":true}\n```'},'done':True})
    client=httpx.AsyncClient
    monkeypatch.setattr(ollama.httpx,'AsyncClient',lambda **kw:client(transport=httpx.MockTransport(handler),**kw))
    monkeypatch.setattr(ollama,'settings',lambda:dict(base_url='http://local.test:11434',model='qwen-test',timeout_seconds=10))
    content=[dict(type='input_text',text='test')]
    if image: content.append(dict(type='input_image',image_url='data:image/png;base64,YWJj'))
    text=asyncio.run(ollama.complete_ollama('instructions',content,{'type':'object'}))
    assert json.loads(text)=={'ok':True}
    req=captured[0];body=json.loads(req.content)
    assert str(req.url)=='http://local.test:11434/api/chat'
    assert 'authorization' not in req.headers
    assert body['stream'] is False and body['format']=={'type':'object'}
    assert body['messages'][1].get('images')==(['YWJj'] if image else None)

@pytest.mark.parametrize('payload',[{'message':{'content':''}},{'error':'failed'},{'message':{'content':'cut'},'done_reason':'length'}])
def test_ollama_rejects_bad_response(monkeypatch,payload):
    client=httpx.AsyncClient
    monkeypatch.setattr(ollama.httpx,'AsyncClient',lambda **kw:client(transport=httpx.MockTransport(lambda r:httpx.Response(200,json=payload)),**kw))
    with pytest.raises(ValueError): asyncio.run(ollama.complete_ollama('test',[]))

@pytest.mark.parametrize('provider', ['ollama', 'openai'])
def test_high_dimensional_evidence_keeps_aligned_coefficients(monkeypatch, provider):
    from backend import app as api
    monkeypatch.setenv('ULCA_LLM_PROVIDER',provider)
    result={
      'intent':{},'variables':[f'Pixel(1,{i+1})' for i in range(60)],
      'loadings':[[i/100,-i/100] for i in range(60)],
      'coordinates':[[.123456,2.123456]],'labels':[0], 'clusters':[],
      'comparisons':[{'features':[{'name':f'Pixel(1,{i+1})','separation_contribution':i/100} for i in range(60)]}],
      'initial_metrics':{},'metrics':{}}
    selected=api.explanation_evidence(result)
    assert 'coordinates' not in selected
    assert selected['clusters'] == result['clusters']
    assert len(selected['variables'])==12
    for name,loading in zip(selected['variables'],selected['loadings']):
        assert loading==result['loadings'][result['variables'].index(name)]
    assert len(result['variables'])==60
