import importlib

import numpy as np
import pytest
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

from backend.engine import (
    Intent, EVDULCA, classify, evaluate, optimize, explain, run_analysis,
)
from backend.dataset import NAME


def test_variance_ratio_uses_each_group_center_and_equal_group_weights():
    y = np.repeat([0, 1, 2], [4, 6, 8])
    rng = np.random.default_rng(4)
    z = rng.normal(size=(len(y), 2))
    intent = Intent("diversity", [0]).validate()
    metrics = evaluate(z, y, intent)
    variances = [np.var(z[y == g], axis=0).sum() for g in range(3)]
    assert metrics["score"] == pytest.approx(variances[0] / (np.mean(variances[1:]) + 1e-8))
    z[y == 1] += [1000, -500]
    z[y == 2] += [-1000, 700]
    assert evaluate(z, y, intent)["score"] == pytest.approx(metrics["score"])
    assert not evaluate(z, y, intent, variances[0] * 2)["feasible"]
    assert evaluate(z, y, intent, variances[0])["feasible"]


@pytest.mark.skipif(NAME != "wine", reason="Wine names")
@pytest.mark.parametrize("prompt", ["Baroloの内部構造の特性が見たい", "Baloroの内部構造が見たい", "赤の多様性を見たい"])
def test_internal_intent(prompt):
    intent = classify(prompt)
    assert (intent.pattern, intent.targets) == ("diversity", [0])
    assert classify("Baroloの特徴を教えて").pattern == "target"


def test_search_preserves_target_spread_and_keeps_feasible_incumbent():
    raw = load_wine()
    x = StandardScaler().fit_transform(raw.data)
    intent = Intent("diversity", [0])
    result = optimize(x, raw.target, intent, budget=60)
    initial_z = EVDULCA().fit_transform(x, raw.target, **result["initial_parameters"])
    z = EVDULCA().fit_transform(x, raw.target, **result["parameters"])
    initial = evaluate(initial_z, raw.target, intent)
    final = evaluate(z, raw.target, intent, initial["target_variance"])
    assert final["feasible"]
    assert final["score"] >= initial["score"] - 1e-10
    assert final["score"] == pytest.approx(result["history"][-1]["best_score"])
    assert len(result["history"]) == 60
    assert result["parameters"]["w_bw"] == [0., 0., 0.]
    assert all(np.diff([h["best_score"] for h in result["history"]]) >= 0)


@pytest.mark.skipif(NAME != "wine", reason="Wine API smoke test")
def test_analysis_and_manual_update_explain_internal_variation():
    from backend.engine import update_projection
    result = run_analysis(Intent("diversity", [0]), budget=15)
    assert result["metrics"]["objective"] == "within_variance_ratio"
    assert result["metrics"]["feasible"]
    assert result["comparisons"] == []
    assert "内部変動" in explain(result)
    updated = update_projection(result, result["parameters"])
    assert updated["metrics"]["score"] == pytest.approx(result["metrics"]["score"])
    assert updated["metrics"]["baseline_target_variance"] == pytest.approx(result["initial_metrics"]["target_variance"])
    api = importlib.import_module("backend.app")
    assert "シルエットではない" in api.explanation_instructions("", "diversity")


@pytest.mark.skipif(NAME != "wine", reason="Wine API")
def test_api_diversity_with_llm_classification(monkeypatch):
    import json
    from fastapi.testclient import TestClient
    api = importlib.import_module("backend.app")
    monkeypatch.setattr(api, "llm_enabled", lambda: True)
    calls = []

    async def fake_complete(instructions, content, schema=None):
        calls.append((instructions, content))
        if schema:
            assert "diversity" in schema["properties"]["pattern"]["enum"]
            return json.dumps({"pattern": "diversity", "targets": [0], "clarification": ""})
        assert "分離の説明ではなく" in instructions
        evidence = json.loads(content[0]["text"])
        assert evidence["comparisons"] == []
        assert evidence["metrics"]["feasible"]
        return "Barolo内の広がりを確認します。Alcohol (0.25)"

    monkeypatch.setattr(api, "complete", fake_complete)
    response = TestClient(api.app).post("/api/analyze", json={"prompt": "Baroloの内部構造の特性が見たい"})
    assert response.status_code == 200
    result = response.json()
    assert result["intent"]["pattern"] == "diversity"
    assert result["trials"] == 200
    assert result["metrics"]["score"] >= result["initial_metrics"]["score"]
    assert result["image_read_by_llm"]
    assert len(calls) == 2
