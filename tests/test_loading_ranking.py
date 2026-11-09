import numpy as np
import pytest

from backend.engine import loading_recommendations, explain, explain_with_llm


def fixture_result(pattern="diversity", direction="横方向"):
    return {
        "intent": {"pattern": pattern, "targets": [0]},
        "variables": ["Small", "Negative", "Positive", "Tie", "Rounding"],
        "loadings": [[.01, -.95], [-.9, .1], [.8001, -.8], [.8001, .2], [.8002, -.1]],
        "coordinates": [[0., 0.], [2., .1], [5., 1.]], "labels": [0, 0, 1],
        "comparisons": [{"direction": direction, "group_a": "A", "group_b": "B"}],
    }


def test_internal_variation_ranks_negative_coefficients_and_each_axis():
    r = fixture_result()
    ranked = loading_recommendations(r)
    assert [f["name"] for f in ranked[0]["features"]] == ["Negative", "Rounding"]
    assert [f["name"] for f in ranked[1]["features"]] == ["Small", "Positive"]
    assert ranked[0]["features"][0]["loading"] == -.9
    text = explain(r)
    assert text.index("Negative (-0.90)") < text.index("Rounding (0.80)")
    assert "Small (-0.95)" in text


@pytest.mark.parametrize("direction,axes", [("横方向", ["X"]), ("縦方向", ["Y"]), ("斜め方向", ["X", "Y"])])
def test_separation_uses_only_corresponding_axes(direction, axes):
    assert [x["axis"] for x in loading_recommendations(fixture_result("target", direction))] == axes


def test_ranking_precedes_rounding_and_ties_keep_original_order():
    r = fixture_result("target")
    r["loadings"][1][0] = 0.
    names = [x["name"] for x in loading_recommendations(r)[0]["features"]]
    assert names == ["Rounding", "Positive", "Tie"]


def test_llm_cannot_override_ranked_names_or_coefficients():
    r = fixture_result()
    text = explain_with_llm(r, "1. 散布図の状況：対象群が横に広がっています。\n\n2. 寄与した変数：Small (9.99)")
    assert "9.99" not in text
    assert "対象群が横に広がっています。" in text
    assert text == explain_with_llm(r, "対象群が横に広がっています。")
    assert text == explain_with_llm(r, "- 散布図の状況：対象群が横に広がっています。\n- 寄与した変数：Small (9.99)")
    # A model ignoring the contract even in its situation cannot leak feature claims.
    text = explain_with_llm(r, "Small (9.99) が最大です。")
    assert "9.99" not in text
    assert text == explain(r)
