"""Explicit manual integration check against the configured local LLM."""
import asyncio,json,time,sys
import httpx
from backend.app import classify_with_llm,knowledge,complete,explanation_evidence,explanation_instructions
from backend.dataset import dataset_context
from backend.engine import run_analysis,Intent
from backend.prompts import EXPLANATION_INSTRUCTIONS

async def main():
    docs=knowledge()
    start=time.perf_counter()
    if '--image-only' not in sys.argv:
        intent=await classify_with_llm('数字6と9を比較して',docs)
        assert intent.pattern=='pair' and intent.targets==[1,2],intent
        print('Classification: pair [1,2]; seconds:',round(time.perf_counter()-start,1),flush=True)
    result=run_analysis(Intent('overall',[]),optimize_parameters=False)
    evidence=explanation_evidence(result)
    explanation=await complete(explanation_instructions(docs),[
        {'type':'input_text','text':json.dumps(evidence,ensure_ascii=False)},
        {'type':'input_image','image_url':result['images']['scatter']}])
    print('Image + numerical explanation:',explanation,flush=True)
    print('Total seconds:',round(time.perf_counter()-start,1),flush=True)

if __name__=='__main__':
    try:asyncio.run(main())
    except httpx.HTTPStatusError as e:
        print('LLM API error:',e.response.text[:2000],flush=True)
        raise
