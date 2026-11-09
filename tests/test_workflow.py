import importlib
import json
import numpy as np
import pytest
from backend.dataset import NAME

pytestmark = pytest.mark.skipif(NAME != "wine", reason="Wine regression suite: set ULCA_DATASET=wine")
from fastapi.testclient import TestClient
from sklearn.metrics import silhouette_samples
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

from backend.engine import (Intent, classify, evaluate, optimize, decode, encode,
    explain, separation_evidence, initial_parameters, EVDULCA)

api = importlib.import_module("backend.app")


@pytest.mark.parametrize("prompt,pattern,targets", [
    ("全体構造が分かるように全クラスタを分離して", "overall", []),
    ("クラスタ０の特徴が分かるように分離して", "target", [0]),
    ("クラスタ０の特徴を教えて", "target", [0]),
    ("赤い点の特徴を教えて", "target", [0]),
    ("Baroroの特徴を教えて", "target", [0]),
    ("BAROROを分離して", "target", [0]),
    ("Baroloの特徴を教えて", "target", [0]),
    ("バローロの特徴を教えて", "target", [0]),
    ("Baroroと緑の点を比較して", "pair", [0, 1]),
    ("クラス1と2を比較して", "pair", [1, 2]),
    ("赤と緑の違い", "pair", [0, 1]),
    ("紫の特徴を見たい", "target", [2]),
])
def test_classification(prompt, pattern, targets):
    result = classify(prompt)
    assert (result.pattern, result.targets) == (pattern, targets)


@pytest.mark.parametrize("prompt", ["クラスの違い", "こんにちは", "クラスタ9の特徴", "クラスタ-1を比較",
                                   "特定2クラスタを比較して", "クラス0と9を比較"])
def test_unknown_scope_requires_clarification(prompt):
    with pytest.raises(ValueError):
        classify(prompt)


@pytest.mark.parametrize("intent", [Intent("overall", []), Intent("target", [1]), Intent("pair", [0, 2])])
def test_objective_exactly_matches_requested_groups(intent):
    # Unequal group sizes distinguish macro averaging from the sample-weighted mean.
    rng = np.random.default_rng(5)
    y = np.repeat([0, 1, 2], [5, 9, 13])
    Z = rng.normal(size=(len(y), 2))+y[:, None]*2
    result = evaluate(Z, y, intent)
    mask = np.isin(y, intent.targets) if intent.pattern == "pair" else np.ones(len(y), dtype=bool)
    labels = (y != 1).astype(int) if intent.pattern == "target" else y[mask]
    samples = silhouette_samples(Z[mask], labels)
    expected = np.mean([samples[labels == c].mean() for c in np.unique(labels)])
    assert result["score"] == pytest.approx(expected)
    if intent.pattern == "pair":
        Z[y == 1] += 1000
        assert evaluate(Z, y, intent)["score"] == pytest.approx(expected)


@pytest.mark.parametrize("intent", [Intent("overall", []), Intent("target", [0]), Intent("pair", [0, 1])])
def test_search_keeps_incumbent_and_contributions_reconstruct_distance(intent):
    raw = load_wine()
    X = StandardScaler().fit_transform(raw.data)
    y = raw.target
    result = optimize(X, y, intent, budget=80)
    model = EVDULCA()
    Z = model.fit_transform(X, y, **result["parameters"])
    assert result["trials"] == 80
    assert np.all(np.diff([h["best_score"] for h in result["history"]]) >= 0)
    assert evaluate(Z, y, intent)["score"] >= result["history"][0]["score"]-1e-12
    assert evaluate(Z, y, intent)["score"] == pytest.approx(result["history"][-1]["best_score"])
    p = result["parameters"]
    for name in ["w_tg", "w_bg", "w_bw"]:
        assert np.all((np.array(p[name]) >= 0) & (np.array(p[name]) <= 1))
    for comp in separation_evidence(X, Z, model.M, y, intent):
        assert sum(f["separation_contribution"] for f in comp["features"]) == pytest.approx(comp["distance"])


def test_weights_are_independent_in_both_directions():
    p = {"w_tg": [.4, .6, .8], "w_bg": [.3, .5, .7], "w_bw": [.2, .4, .9], "alpha": 12.}
    x = encode(p)
    assert len(x) == 10
    assert decode(x) == p
    for k in range(3):
        tg_changed = x.copy()
        tg_changed[k] = .9
        assert decode(tg_changed)["w_bg"] == p["w_bg"]
        assert decode(tg_changed)["w_tg"][k] == .9
        bg_changed = x.copy()
        bg_changed[3+k] = .95
        assert decode(bg_changed)["w_tg"] == p["w_tg"]
        assert decode(bg_changed)["w_bg"][k] == .95


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("ULCA_LLM_PROVIDER", "openai")
    monkeypatch.delenv("PROTOTYPE_OPENAI_KEY", raising=False)
    monkeypatch.delenv("PROTOTYPE_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with TestClient(api.app) as client:
        yield client


@pytest.mark.parametrize("prompt,pattern", [
    ("全体構造を見たい", "overall"),
    ("クラスタ0の特徴を見たい", "target"),
    ("クラスタ０の特徴を教えて", "target"),
    ("クラスタ0と1を比較して", "pair"),
])
def test_full_workflow(client, prompt, pattern):
    response = client.post("/api/analyze", json={"prompt": prompt})
    assert response.status_code == 200
    r = response.json()
    assert r["intent"]["pattern"] == pattern
    assert r["trials"] == 200
    assert r["metrics"]["score"] >= r["initial_metrics"]["score"]-1e-12
    assert len(r["coordinates"]) == 178
    assert len(r["loadings"]) == len(r["variables"]) == 13
    assert r["variables"][0] == "Alcohol" and r["variables"][-1] == "Proline"
    assert r["image_read_by_llm"] is False
    assert all(image.startswith("data:image/png;base64,") for image in r["images"].values())
    assert r["comparisons"][0]["features"][0]["name"] == "Alcohol"
    assert r["explanation"] and r["classification_source"] == "ルール"
    assert all(name in {"Barolo", "Grignolino", "Barbera", "その他"}
               for name in r["metrics"]["groups"])
    assert "クラスタ0" not in r["explanation"]


def test_api_validates_and_serves_ui(client):
    assert client.get("/").status_code == 200
    assert client.get("/assets/app.js").status_code == 200
    page = client.get("/").text
    assert 'id="budget"' not in page
    assert 'id="loading-order"' in page
    assert 'value="magnitude">寄与度順' in page
    assert client.get("/api/status").json()["llm_enabled"] is False
    assert client.post("/api/analyze", json={"prompt": "クラスの特徴"}).json()["status"] == "clarification"
    assert client.post("/api/analyze", json={"prompt": "全体", "budget": 5000}).status_code == 422


def test_initial_ui_uses_current_assets_and_default_projection(client):
    page = client.get("/")
    assert 'id="projection-mode"' not in page.text
    for path in ("/", "/assets/app.js", "/assets/style.css", "/api/initial"):
        response = client.get(path)
        assert response.status_code == 200
        assert "no-store" in response.headers["cache-control"]
    result = client.get("/api/initial").json()
    assert result["parameters"] == {"w_tg": [0, 0, 0], "w_bg": [1, 1, 1],
                                     "w_bw": [1, 1, 1], "alpha": 10}
    assert result["trials"] == 0
    assert len(result["coordinates"]) == 178


@pytest.mark.parametrize("direction,expected", [("横方向", "Alcohol (-0.12)"), ("縦方向", "Alcohol (0.68)"), ("斜め方向", "Alcohol (-0.12)")])
def test_explanation_displays_raw_loadings(direction, expected):
    result = {
        "intent": {"comparison": "クラスタ0 対 クラスタ1"},
        "variables": ["Alcohol", "Ash"], "loadings": [[-.1234, .6789], [.3, .4]],
        "comparisons": [{
            "group_a": "クラスタ0", "group_b": "クラスタ1", "direction": direction,
            "features": [
                {"name": "Alcohol", "separation_contribution": .4273},
                {"name": "Ash", "separation_contribution": -.85},
            ],
        }],
    }
    text = explain(result)
    assert expected in text
    assert "0.427" not in text
    assert "括弧内は" not in text
    assert "Ash" in text
    if direction == "横方向":
        assert text.index("Ash (0.30)") < text.index("Alcohol (-0.12)")
        assert "Y軸：" not in text


def test_llm_receives_copied_knowledge_and_same_plot(client, monkeypatch):
    monkeypatch.setenv("PROTOTYPE_OPENAI_KEY", "test-only")
    monkeypatch.setenv("PROTOTYPE_OPENAI_MODEL", "test-model")
    calls = []
    from types import SimpleNamespace
    clock = iter([0., 1., 3., 10., 13., 15.])
    monkeypatch.setattr(api, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    async def fake_complete(instructions, content, schema=None):
        calls.append((instructions, content, schema))
        if schema:
            return json.dumps({"pattern": "pair", "targets": [0, 1], "clarification": ""})
        return "画像と数値を参照した説明です。"
    monkeypatch.setattr(api, "complete", fake_complete)
    r = client.post("/api/analyze", json={"prompt": "赤と緑の違い"}).json()
    assert r["image_read_by_llm"] is True
    assert r["classification_source"] == "LLM"
    timing = r["llm_timing"]
    for key in ["classification", "explanation"]:
        assert timing[key]["status"] == "complete"
    assert timing["classification"]["seconds"] == 2.
    assert timing["explanation"]["seconds"] == 3.
    assert timing["total_seconds"] == 5.
    assert r["total_seconds"] == 15.
    assert timing["total_seconds"] == pytest.approx(sum(timing[k]["seconds"] for k in ["classification", "explanation"]))
    assert timing["total_seconds"] < r["total_seconds"]
    assert "WORKFLOW.md" in calls[0][0] and "ULCA.md" in calls[0][0]
    assert calls[1][1][1]["image_url"] == r["images"]["scatter"]
    evidence = json.loads(calls[1][1][0]["text"])
    # Point coordinates are supplied visually; numeric evidence retains class geometry.
    assert "coordinates" not in evidence
    assert evidence["clusters"] == r["clusters"]
    assert evidence["loadings"] == r["loadings"]
    assert "クラス番号やクラスタ番号を名前として使わない" in calls[1][0]


def test_llm_failure_is_visible_and_preserves_results(client, monkeypatch):
    import httpx
    monkeypatch.setenv("PROTOTYPE_OPENAI_KEY", "test-only")
    monkeypatch.setenv("PROTOTYPE_OPENAI_MODEL", "test-model")
    async def failed(*args, **kwargs):
        raise httpx.ConnectError("Test failure")
    monkeypatch.setattr(api, "complete", failed)
    r = client.post("/api/analyze", json={"prompt": "クラス0を分離"}).json()
    assert r["status"] == "complete"
    assert r["image_read_by_llm"] is False and len(r["warnings"]) == 2
    assert all(r["llm_timing"][k]["status"] == "failed" for k in ["classification", "explanation"])
    assert r["llm_timing"]["total_seconds"] > 0


def test_manual_parameters_and_inverse_geometry_adjustment(client):
    result = client.post("/api/analyze", json={
        "prompt": "クラスタ0と1を比較して"
    }).json()
    payload = {
        "pattern": result["intent"]["pattern"],
        "targets": result["intent"]["targets"],
        "parameters": result["parameters"],
        "reference_coordinates": result["coordinates"],
        "initial_coordinates": result["initial_coordinates"],
        "history": result["history"],
    }
    payload["parameters"]["w_tg"][0] = .25
    updated = client.post("/api/update", json=payload)
    assert updated.status_code == 200
    manual = updated.json()
    assert manual["parameters"]["w_tg"][0] == pytest.approx(.25)
    assert len(manual["coordinates"]) == 178

    geometry = []
    for cluster in manual["clusters"]:
        variance = cluster["variance"]
        geometry.append({
            "label": cluster["class"],
            "center": cluster["centroid"],
            "area": max(1e-8, np.sqrt(variance[0] * variance[1])),
        })
    geometry[0]["center"][0] += .5
    inverse_payload = {
        **payload,
        "parameters": manual["parameters"],
        "reference_coordinates": manual["coordinates"],
        "geometry": geometry,
        "updated_label": 0,
        "interaction": "move",
        "max_iterations": 10,
    }
    inverse = client.post("/api/inverse", json=inverse_payload)
    assert inverse.status_code == 200
    adjusted = inverse.json()
    assert len(adjusted["coordinates"]) == 178
    assert 0 <= adjusted["parameters"]["alpha"] <= 20
    assert adjusted["inverse_adjustment"]["iterations"] <= 10
    assert adjusted["inverse_adjustment"]["objective"] <= adjusted["inverse_adjustment"]["objective_before"]
    assert all(0 <= weight <= 1 for key in ("w_tg", "w_bg", "w_bw")
               for weight in adjusted["parameters"][key])


def test_manual_parameters_reject_out_of_range_weights(client):
    result = client.post("/api/analyze", json={
        "prompt": "クラスタ0と1を比較して"
    }).json()
    response = client.post("/api/update", json={
        "pattern": "pair", "targets": [0, 1],
        "parameters": {**result["parameters"], "w_bg": [1.1, 0, 0]},
        "reference_coordinates": result["coordinates"],
        "initial_coordinates": result["initial_coordinates"],
    })
    assert response.status_code == 422


def test_box_plot_data_matches_projection_samples(client):
    data = client.get("/api/dataset").json()
    wine = load_wine()
    assert data["labels"] == wine.target.tolist()
    np.testing.assert_allclose(data["raw"], wine.data, rtol=0, atol=0)
    assert data["variables"][0] == "Alcohol"
    assert len(data["variables"]) == wine.data.shape[1]


def test_coincident_centroids_are_not_described_as_separated():
    result = {
        "intent": {"comparison": "Barolo 対 Grignolino"},
        "comparisons": [{"group_a": "Barolo", "group_b": "Grignolino",
                         "direction": "重心が一致", "features": []}],
    }
    text = explain(result)
    assert "重心は一致しています" in text
    assert "分かれて" not in text
