"""Wine analysis API and static UI, served through backend.main on port 8001."""
import asyncio
import json
import math
import os
from pathlib import Path
import time
from typing import Literal

from dotenv import load_dotenv
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .dataset import load_dataset, FEATURES, LABELS, DISPLAY, NAME, dataset_context
from starlette.concurrency import run_in_threadpool

from .prompts import EXPLANATION_INSTRUCTIONS
from .ollama_client import settings as llm_settings, complete_ollama

from .engine import (Intent, PATTERNS, classify, inverse_adjust, run_analysis,
                     update_projection, explain, explain_with_llm, loading_recommendations, ProjectionFailure)

ROOT = Path(__file__).resolve().parent
# Load environment variables from backend/.env or workspace root
load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env")

app = FastAPI(title="Fujiwara ULCA Agent")
FRONTEND = ROOT.parent / "frontend"
app.mount("/assets", StaticFiles(directory=FRONTEND/"public"/"assets"), name="assets")
analysis_lock = asyncio.Lock()


@app.exception_handler(ProjectionFailure)
async def projection_failure_handler(request, error):
    return JSONResponse(status_code=422, content={"status": "failed", "detail": str(error)})


@app.middleware("http")
async def disable_ui_cache(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/assets/") or request.url.path == "/api/initial":
        response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


class Request(BaseModel):
    prompt: str = Field(min_length=1, max_length=3000)
    budget: Literal[200] = 200
    seed: int = Field(default=17, ge=0, le=2147483647)


class Parameters(BaseModel):
    w_tg: list[float] = Field(min_length=3, max_length=3)
    w_bg: list[float] = Field(min_length=3, max_length=3)
    w_bw: list[float] = Field(min_length=3, max_length=3)
    alpha: float = Field(ge=0, le=20)


class AnalysisEdit(BaseModel):
    pattern: str
    targets: list[int]
    parameters: Parameters
    reference_coordinates: list[list[float]]
    initial_coordinates: list[list[float]]
    history: list[dict] = Field(default_factory=list)


class TargetGeometry(BaseModel):
    label: int = Field(ge=0, le=2)
    center: list[float] = Field(min_length=2, max_length=2)
    area: float = Field(gt=0)


class InverseEdit(BaseModel):
    pattern: str
    targets: list[int]
    parameters: Parameters
    geometry: list[TargetGeometry] = Field(min_length=3, max_length=3)
    updated_label: int = Field(ge=0, le=2)
    interaction: Literal["move", "scale"]
    max_iterations: int = Field(default=50, ge=10, le=100)
    reference_coordinates: list[list[float]]
    initial_coordinates: list[list[float]]
    history: list[dict] = Field(default_factory=list)


def checked_intent(pattern, targets):
    try:
        return Intent(pattern, targets).validate()
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def checked_parameters(parameters):
    values = parameters.model_dump()
    if any(not 0 <= weight <= 1 for key in ("w_tg", "w_bg", "w_bw")
           for weight in values[key]):
        raise HTTPException(status_code=422, detail="各重みは0〜1で指定してください。")
    if not math.isfinite(values["alpha"]):
        raise HTTPException(status_code=422, detail="alphaには有限値を指定してください。")
    return values


def analysis_state(pattern, targets, parameters, reference_coordinates, initial_coordinates, history):
    intent = checked_intent(pattern, targets)
    for name, coordinates in (("reference", reference_coordinates),
                              ("initial", initial_coordinates)):
        if len(coordinates) != len(load_dataset().target) or any(len(point) != 2 for point in coordinates):
            raise HTTPException(status_code=422, detail=f"{name}座標は{len(load_dataset().target)}点の2次元で指定してください。")
        if any(not math.isfinite(value) for point in coordinates for value in point):
            raise HTTPException(status_code=422, detail="座標には有限値を指定してください。")
    return {
        "intent": {"pattern": intent.pattern, "targets": intent.targets,
                   "name": PATTERNS[intent.pattern], "comparison": intent.description()},
        "labels": load_dataset().target.tolist(),
        "coordinates": reference_coordinates,
        "initial_coordinates": initial_coordinates,
        "history": history,
        "parameters": checked_parameters(parameters),
    }


def get_openai_credentials():
    key = os.environ.get("PROTOTYPE_OPENAI_KEY") or os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("PROTOTYPE_OPENAI_MODEL") or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return key, model


def llm_enabled():
    config = llm_settings()
    if config['provider'] == 'ollama':
        return bool(config['base_url'] and config['model'])
    if config['provider'] != 'openai':
        return False
    key, model = get_openai_credentials()
    return bool(key and model)


def knowledge():
    # Local copies only. WORKFLOW.md explicitly supersedes older heuristic wording.
    return "\n\n".join(f"## {name}\n{(ROOT/'knowledge'/name).read_text(encoding='utf-8')}"
        for name in ["ULCA.md", "Parameter_Tuning.md", "Interpretation.md", "WORKFLOW.md"])


async def complete(instructions, content, schema=None):
    if llm_settings()['provider'] == 'ollama':
        return await complete_ollama(instructions, content, schema)
    if llm_settings()['provider'] != 'openai':
        raise ValueError('LLM provider is disabled')
    key, model = get_openai_credentials()
    body = {"model": model, "store": False,
            "instructions": instructions, "input": [{"role": "user", "content": content}]}
    if schema:
        body["text"] = {"format": {"type": "json_schema", "name": "analysis_intent", "strict": True, "schema": schema}}
    async with httpx.AsyncClient(timeout=60) as client:
        for attempt in range(3):
            try:
                response = await client.post("https://api.openai.com/v1/responses", json=body,
                    headers={"Authorization": f'Bearer {key}'})
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as error:
                try:
                    code = error.response.json().get("error", {}).get("code")
                except (ValueError, AttributeError):
                    code = None
                if attempt == 2 or error.response.status_code not in (408, 429, 500, 502, 503, 504) or code == "insufficient_quota":
                    raise
            except (httpx.TransportError, httpx.TimeoutException):
                if attempt == 2:
                    raise
            await asyncio.sleep(.5 * 2**attempt)
        payload = response.json()
    text = "".join(part.get("text", "") for item in payload.get("output", [])
                   for part in item.get("content", []) if part.get("type") == "output_text")
    if not text.strip():
        raise ValueError("Empty model response")
    return text


def llm_failure_reason(error):
    """Expose useful diagnostics without request bodies, credentials or raw errors."""
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        try:
            code = error.response.json().get("error", {}).get("code")
        except (ValueError, AttributeError):
            code = None
        if code == "insufficient_quota":
            return "APIの利用可能残高・利用上限を確認してください（HTTP 429）。"
        reason = {401: "API認証に失敗", 403: "APIへのアクセスが拒否", 429: "APIの利用制限", 400: "APIがリクエストを受理できません", 404: "APIのモデルまたは接続先が見つかりません"}.get(status, "APIサーバーのエラー")
        return f"{reason}（HTTP {status}）。"
    if isinstance(error, httpx.TimeoutException):
        return "LLMの応答待ちがタイムアウトしました。"
    if isinstance(error, httpx.TransportError):
        return "LLMへの通信に失敗しました。ネットワーク接続を確認してください。"
    return "LLMの応答形式を読み取れませんでした。"


async def classify_with_llm(prompt, docs):
    schema = {"type": "object", "properties": {
        "pattern": {"type": "string", "enum": ["overall", "target", "pair", "diversity", "clarify"]},
        "targets": {"type": "array", "items": {"type": "integer"}},
        "clarification": {"type": "string"}},
        "required": ["pattern", "targets", "clarification"], "additionalProperties": False}
    instructions = (
        "ULCA分析の意図を分類してください。利用可能なのはWineの0=Barolo/赤、1=Grignolino/緑、2=Barbera/紫。"
        "色や品種名でも対象を特定する。『赤い点』『赤のクラスタ』『Barolo』『Baroro』『バローロ』はすべてクラスタ0を指す。"
        "BaroroはBaroloの別表記として受け入れ、綴りの確認は不要。英字の大文字・小文字は区別しない。"
        "overallは全体構造(targets=[]), targetは1クラスタ対その他(targets=[対象]), pairは2クラスタ比較。"
        "diversityは特定1群の内部構造・内部変動・多様性・群内のばらつきの探索(targets=[対象])。"
        "例：『Baroloの内部構造の特性が見たい』はdiversity。BaloroもBaroloの表記揺れとして扱う。"
        "内部構造や多様性の明示は通常の特徴把握targetより優先する。"
        "特定クラスタの『特徴を教えて』『特徴を知りたい』『特徴を見たい』も、そのクラスタをその他から分離する最適化の依頼としてtargetに分類する。"
        "『分離』『最適化』という明示がなくても確認質問は不要。例：『Baroloの特徴を教えて』はpattern=target, targets=[0], clarification=空文字。"
        "対象不明・存在しない番号・一般会話・矛盾した指示はclarifyとし、質問をclarificationに記述。"
        "ユーザーの指示に全体が明示されていない限り、対象不明をoverallに置き換えない。"
        "次の知識のうちWORKFLOW.mdが今回の実装仕様として優先されます。\n"+docs)
    instructions += "\n" + dataset_context()
    try:
        data = json.loads(await complete(instructions, [{"type": "input_text", "text": prompt}], schema))
        if not isinstance(data, dict) or data.get("pattern") not in ["overall", "target", "pair", "diversity", "clarify"] or not isinstance(data.get("targets"), list):
            raise ValueError("Invalid classification")
        if data["pattern"] != "clarify":
            Intent(data["pattern"], data["targets"]).validate()
    except ValueError as error:
        raise json.JSONDecodeError("Invalid LLM classification", "", 0) from error
    if data["pattern"] == "clarify":
        raise ValueError(data["clarification"] or "比較するクラスタを指定してください。")
    return Intent(data["pattern"], data["targets"]).validate()


@app.get("/")
def index():
    return FileResponse(FRONTEND/"index.html")


@app.get("/api/dataset")
def dataset():
    raw = load_dataset()
    return {"id": NAME, "name": DISPLAY, "variables": FEATURES,
            "class_names": LABELS, "labels": raw.target.tolist(), "raw": raw.data.tolist()}


@app.get("/api/status")
def status():
    return {"llm_enabled": llm_enabled(), "dataset": DISPLAY, "independent": False, "llm_provider": llm_settings()["provider"],
            "llm_model": llm_settings()["model"] if llm_settings()["provider"] == "ollama" else get_openai_credentials()[1],
            "knowledge": ["ULCA.md", "Parameter_Tuning.md", "Interpretation.md", "WORKFLOW.md"]}


@app.get("/api/initial")
async def initial():
    async with analysis_lock:
        result = await run_in_threadpool(
            run_analysis, Intent("overall", []), optimize_parameters=False)
    result.update(status="complete", prompt="初期表示", warnings=[],
                  explanation="", explanation_source="初期設定", image_read_by_llm=False)
    return result


def explanation_evidence(result):
    # The scatter image already carries the point layout. Sending every point's
    # coordinates and label again adds thousands of redundant values to the
    # vision request without improving the numeric explanation.
    keys = ["intent", "variables", "clusters", "loadings", "comparisons", "initial_metrics", "metrics"]
    evidence = {k: result[k] for k in keys}
    evidence["recommended_variables"] = loading_recommendations(result)
    if llm_settings()["provider"] != "ollama" and len(result["variables"]) <= 50:
        return evidence
    selected = set()
    for axis in [0, 1]:
        selected.update(sorted(range(len(result["variables"])),
                               key=lambda i: abs(result["loadings"][i][axis]), reverse=True)[:12])
    comparisons = []
    for item in result["comparisons"]:
        features = sorted(item["features"], key=lambda f: abs(f["separation_contribution"]), reverse=True)[:12]
        selected.update(result["variables"].index(f["name"]) for f in features)
        comparisons.append(dict(item, features=features))
    indices = sorted(selected)
    evidence = dict(evidence, variables=[result["variables"][i] for i in indices],
                    loadings=[result["loadings"][i] for i in indices], comparisons=comparisons,
                    variable_selection="Top 12 absolute loadings per axis and top 12 absolute separation contributions per comparison; point layout is provided by the scatter image")
    def rounded(value):
        if isinstance(value, float): return round(value, 4)
        if isinstance(value, list): return [rounded(v) for v in value]
        if isinstance(value, dict): return {k: rounded(v) for k, v in value.items()}
        return value
    return rounded(evidence)


def explanation_instructions(docs, pattern=None):
    if pattern == "diversity":
        text = (
            "対象群の内部変動を探索したULCAの散布図と数値を日本語で説明してください。"
            "群間分離の説明ではなく、対象群のX・Y方向の広がりと各軸を構成する変数を説明する。"
            "各軸のloadingsの絶対値上位2〜3変数を元の変数名と符号付き小数2桁の係数で示す。"
            "係数だけから対象群内の分散への個別寄与、元の値の高低、因果関係を断定しない。"
            "サブクラスタの存在を断定しない。metricsのscoreは分散比でありシルエットではない。"
            "目的は対象群内分散を初期以上に保ちながら、対象群内分散/他の各群の群内分散の平均を最大化すること。"
            "このモードの指示は以下の分離向けの記述より優先する。\n" + docs)
        text += (
            "\n【内部変動モードの出力形式：最優先】\n"
            "他の分析と同じく、次の2項目だけを日本語で簡潔に返す。全体で300文字程度以内。\n"
            "1. 散布図の状況：対象群がどの方向に広がっているかを1文で述べる。\n"
            "2. 寄与した変数：X軸・Y軸それぞれの変動を構成する変数を、"
            "loadings絶対値順で上位2個ずつ、元の変数名と符号付き小数2桁の係数で示す。各軸1行。\n"
            "前置き、分析目的の復唱、結論、考察、注意書き、指標の説明、"
            "分散や分散比の数値、最適化前後の比較は出力しない（分析詳細に表示済み）。"
            "係数を寄与率と呼ばず、群間分離の説明へ置き換えない。")
    else:
        text = EXPLANATION_INSTRUCTIONS + docs
    if NAME == "mnist069":
        for source, target in [("Barolo", "Digit 0"), ("Grignolino", "Digit 6"),
                               ("Barbera", "Digit 9"), ("各品種", "各数字")]:
            text = text.replace(source, target)
    return text + "\n" + dataset_context() + (
        "\n【今回の応答契約：最優先】散布図の状況だけを1〜2文で返す。"
        "見出し、変数名、係数、寄与した変数の節は出力しない。"
        "寄与した変数はプログラムがrecommended_variablesの順序・数値で追記する。"
        "横方向のみの分離ならY軸を分離の根拠としない。")


@app.post("/api/analyze")
async def analyze(req: Request):
    started = time.perf_counter()
    warnings = []
    docs = knowledge()
    classification_source = "ルール"
    llm_timing = {"classification": {"seconds": 0., "status": "skipped"},
                  "explanation": {"seconds": 0., "status": "skipped"}}
    try:
        if llm_enabled():
            classification_started = time.perf_counter()
            llm_timing["classification"]["status"] = "failed"
            try:
                intent = await classify_with_llm(req.prompt, docs)
                classification_source = "LLM"
                llm_timing["classification"]["status"] = "complete"
            except (httpx.HTTPError, json.JSONDecodeError, KeyError) as error:
                llm_timing["classification"]["error"] = llm_failure_reason(error)
                warnings.append("LLM分類に失敗したため、ルールで分類しました。" + llm_failure_reason(error))
                intent = classify(req.prompt)
            finally:
                llm_timing["classification"]["seconds"] = time.perf_counter()-classification_started
        else:
            intent = classify(req.prompt)
    except ValueError as error:
        return {"status": "clarification", "message": str(error)}

    # All calculation state is local to this request; no existing app sessions are touched.
    async with analysis_lock:
        result = await run_in_threadpool(run_analysis, intent, 200, req.seed)
    result["prompt"] = req.prompt
    result["classification_source"] = classification_source
    result["explanation"] = explain(result)
    result["explanation_source"] = ""
    result["image_read_by_llm"] = False
    if llm_enabled():
        # Include coordinates, covariance, original-order loadings and exact contributions.
        evidence = explanation_evidence(result)
        explanation_started = time.perf_counter()
        llm_timing["explanation"]["status"] = "failed"
        try:
            situation = await complete(
                explanation_instructions(docs, intent.pattern),
                [{"type": "input_text", "text": json.dumps(evidence, ensure_ascii=False)},
                 {"type": "input_image", "image_url": result["images"]["scatter"]}])
            result["explanation"] = explain_with_llm(result, situation)
            result["explanation_source"] = ""
            result["image_read_by_llm"] = True
            llm_timing["explanation"]["status"] = "complete"
        except (httpx.HTTPError, ValueError, KeyError) as error:
            llm_timing["explanation"]["error"] = llm_failure_reason(error)
            warnings.append("LLMの画像説明を取得できなかったため、数値根拠による定型説明を表示します。" + llm_failure_reason(error))
        finally:
            llm_timing["explanation"]["seconds"] = time.perf_counter()-explanation_started
    llm_timing["total_seconds"] = sum(llm_timing[k]["seconds"] for k in ["classification", "explanation"])
    result.update(status="complete", warnings=warnings, llm_timing=llm_timing, total_seconds=time.perf_counter()-started)
    return result


@app.post("/api/update")
async def update(req: AnalysisEdit):
    state = analysis_state(req.pattern, req.targets, req.parameters,
                           req.reference_coordinates, req.initial_coordinates, req.history)
    async with analysis_lock:
        result = await run_in_threadpool(update_projection, state, state["parameters"])
    return result


@app.post("/api/inverse")
async def inverse(req: InverseEdit):
    state = analysis_state(req.pattern, req.targets, req.parameters,
                           req.reference_coordinates, req.initial_coordinates, req.history)
    geometry = sorted(req.geometry, key=lambda item: item.label)
    if [item.label for item in geometry] != [0, 1, 2]:
        raise HTTPException(status_code=422, detail="全3クラスタの目標形状が必要です。")
    if any(not math.isfinite(value) for item in geometry
           for value in [*item.center, item.area]):
        raise HTTPException(status_code=422, detail="目標形状には有限値を指定してください。")
    async with analysis_lock:
        result = await run_in_threadpool(
            inverse_adjust, state,
            [{"label": item.label, "center": item.center, "area": item.area}
             for item in geometry],
            req.updated_label, req.interaction, req.max_iterations)
    return result
