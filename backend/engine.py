"""Wine projection, parameter searches, evaluation, and numerical explanations."""

import base64
import io
import itertools
import re
import time
import unicodedata
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from .dataset import load_dataset, FEATURES, LABELS, ALIASES, NAME, DISPLAY
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples
from threadpoolctl import threadpool_limits

from .ulca import EVDULCA

COLORS = ["#b42046", "#21845a", "#6946ad"]
PATTERNS = {"overall": "全体構造の把握", "target": "特定1クラスタの特徴把握", "pair": "特定2クラスタの比較", "diversity": "特定群の内部変動の探索"}


def single_group_pattern(text):
    return "diversity" if any(word in text for word in
        ["内部構造", "内部変動", "多様性", "群内", "ばらつき", "バラツキ", "書き方", "internal", "diversity", "within"]) else "target"



@dataclass
class Intent:
    pattern: str
    targets: list[int]

    def validate(self):
        expected = {"overall": 0, "target": 1, "pair": 2, "diversity": 1}
        if self.pattern not in expected or len(self.targets) != expected[self.pattern]:
            raise ValueError("比較対象を明記してください。例：BaroloとGrignolinoを比較して")
        if any(type(t) is not int or t not in range(3) for t in self.targets) or len(set(self.targets)) != len(self.targets):
            raise ValueError("対象クラスタを確認してください。")
        return self

    def description(self):
        if self.pattern == "overall":
            return "全クラスタ間（各クラスタを等しい重みで評価）"
        if self.pattern == "target":
            return f"{LABELS[self.targets[0]]} 対 その他（その他を一群として評価）"
        if self.pattern == "diversity":
            return f"{LABELS[self.targets[0]]}の内部変動を強調・他の各群を圧縮"
        return f"{LABELS[self.targets[0]]} 対 {LABELS[self.targets[1]]}（他クラスタは評価から除外）"


def classify(prompt):
    text = unicodedata.normalize("NFKC", prompt).lower()
    if NAME == "mnist069":
        digits = re.findall(r"(?<![\d.-])(-?\d+)(?![\d.])", text)
        if any(t in text for t in ["全体", "全クラスタ", "全クラス", "すべて", "全て", "overall"]):
            return Intent("overall", []).validate()
        if any(int(d) not in [0, 6, 9] for d in digits):
            raise ValueError("対象の数字は0・6・9です。")
        targets = { {0: 0, 6: 1, 9: 2}[int(d)] for d in digits }
        targets.update(i for i, aliases in enumerate(ALIASES) if any(a in text for a in aliases))
        if len(targets) in [1, 2]:
            return Intent(single_group_pattern(text) if len(targets) == 1 else "pair", sorted(targets)).validate()
        raise ValueError("全体、数字0・6・9のいずれか、または2つの数字を指定してください。")
    explicit = re.findall(r"(?:クラスター?|クラスタ|クラス|class|cluster)\s*(-?\d+)", text)
    if any(int(t) not in range(3) for t in explicit):
        raise ValueError("対象クラスタを確認してください。")
    targets = {int(t) for t in explicit}
    # Accept paired IDs, but do not interpret the count in "2クラスタを比較" as class 2.
    number_pairs = re.findall(r"(?<![\d.-])(\d+)\s*(?:と|対|vs\.?|and)\s*(\d+)(?![\d.])", text)
    for pair in number_pairs:
        if any(int(t) not in range(3) for t in pair):
            raise ValueError("対象クラスタを確認してください。")
        targets.update(map(int, pair))
    for i, aliases in enumerate(ALIASES):
        if any(a in text for a in aliases):
            targets.add(i)
    if any(s in text for s in ["全体", "全クラスタ", "全クラス", "すべて", "全て", "overall", "all classes"]):
        return Intent("overall", []).validate()
    if len(targets) in [1, 2]:
        return Intent(single_group_pattern(text) if len(targets) == 1 else "pair", sorted(targets)).validate()
    raise ValueError("全体構造・対象1クラスタ・対象2クラスタのどれを分析しますか？対象の名前も指定してください。")


def initial_parameters(intent):
    tg = np.zeros(3)
    bg = np.ones(3)
    bw = np.ones(3)
    if intent.pattern == "diversity":
        bg[:] = .5
        tg[intent.targets] = 1
        bg[intent.targets] = 0
        return dict(w_tg=tg.tolist(), w_bg=bg.tolist(), w_bw=[0.]*3, alpha=10.)
    if intent.pattern != "overall":
        bw[:] = 0
        bw[intent.targets] = 1
        tg[intent.targets] = 1
        bg[intent.targets] = 0
    return dict(w_tg=tg.tolist(), w_bg=bg.tolist(), w_bw=bw.tolist(), alpha=10.)


def encode(p):
    return np.r_[p["w_tg"], p["w_bg"], p["w_bw"], p["alpha"]/20]


def decode(x):
    x = np.asarray(x)
    return dict(w_tg=x[:3].tolist(), w_bg=x[3:6].tolist(),
                w_bw=x[6:9].tolist(), alpha=float(x[9]*20))


class ProjectionFailure(ValueError):
    """The requested two dimensional projection has collapsed."""


def validate_projection(Z):
    # Singular values detect collapse regardless of axis rotation/alignment.
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2 or Z.shape[1] != 2 or len(Z) < 2 or not np.isfinite(Z).all():
        raise ProjectionFailure("失敗：有効な2次元の散布図を得られませんでした。")
    singular = np.linalg.svd(Z - Z.mean(axis=0), compute_uv=False)
    if singular[0] / np.sqrt(len(Z)) <= 1e-8 or singular[1] <= singular[0] * 1e-3:
        raise ProjectionFailure("失敗：散布図が直線状または点に潰れました。αを下げるか、群内ばらつきの重みを調整してください。")


def evaluate(Z, y, intent, baseline_target=None):
    validate_projection(Z)
    if intent.pattern == "diversity":
        groups = {LABELS[int(k)]: float(np.var(Z[y == k], axis=0).sum()) for k in np.unique(y)}
        target = groups[LABELS[intent.targets[0]]]
        others = float(np.mean([v for k, v in groups.items() if k != LABELS[intent.targets[0]]]))
        # Fixed epsilon on standardized input / orthonormal projection coordinates.
        return {"score": target / (others + 1e-8), "groups": groups,
                "objective": "within_variance_ratio", "target_variance": target,
                "other_variance": others, "epsilon": 1e-8,
                "baseline_target_variance": baseline_target,
                "feasible": baseline_target is None or target >= baseline_target - 1e-10}
    mask = np.ones(len(y), dtype=bool)
    labels = y
    if intent.pattern == "target":
        labels = (y != intent.targets[0]).astype(int)
    elif intent.pattern == "pair":
        mask = np.isin(y, intent.targets)
        labels = y[mask]
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2 or len(unique) >= len(labels) or min(counts) < 2:
        raise ValueError("シルエット評価には各群2点以上、2群以上が必要です。")
    samples = silhouette_samples(Z[mask], labels)
    groups = {str(int(k)): float(samples[labels == k].mean()) for k in unique}
    if intent.pattern == "target":
        groups = {LABELS[intent.targets[0]]: groups["0"], "その他": groups["1"]}
    else:
        groups = {LABELS[int(k)]: v for k, v in groups.items()}
    return {"score": float(np.mean(list(groups.values()))), "groups": groups}


def optimize(X, y, intent, budget=200, seed=17):
    """Budgeted multistart coordinate search; return incumbent, never final trial."""
    intent.validate()
    if not 10 <= budget <= 2000:
        raise ValueError("試行回数は10〜2000で指定してください。")
    start = time.perf_counter()
    history = []
    best = -np.inf
    best_x = None
    rng = np.random.default_rng(seed)
    initial = initial_parameters(intent)
    baseline_target = None
    if intent.pattern == "diversity":
        initial_Z = EVDULCA().fit_transform(X, y, **initial)
        baseline_target = evaluate(initial_Z, y, intent)["target_variance"]
    starts = [encode(initial)]
    compact = dict(initial, w_tg=[0., 0., 0.], w_bg=[1., 1., 1.])
    starts.extend([encode(compact), encode(dict(w_tg=[0.]*3, w_bg=[1.]*3, w_bw=[1.]*3, alpha=10.))])

    class Done(Exception):
        pass

    def objective(x):
        nonlocal best, best_x
        if len(history) >= budget:
            raise Done()
        x = np.array(x, dtype=float).copy()
        if intent.pattern == "diversity":
            x[6:9] = 0  # Internal variation mode does not reward centroid separation.
        model = EVDULCA()
        try:
            metrics = evaluate(model.fit_transform(X, y, **decode(x)), y, intent, baseline_target)
            score = metrics["score"]
        except ProjectionFailure as error:
            history.append({"trial": len(history)+1, "score": -1.,
                            "best_score": float(best) if best_x is not None else -1.,
                            "valid": False, "failure": str(error),
                            "seconds": time.perf_counter()-start})
            return -1.
        if not np.isfinite(score):
            raise ValueError("有限の評価値を得られませんでした。")
        feasible = metrics.get("feasible", True)
        if feasible and score > best:
            best, best_x = score, np.array(x).copy()
        history.append({"trial": len(history)+1, "score": score, "best_score": best,
                        "feasible": feasible,
                        "seconds": time.perf_counter()-start})
        return score if feasible else -1.

    with threadpool_limits(limits=1):
        try:
            for x in starts:
                objective(x)
            restart = 0
            while True:
                x = starts[restart].copy() if restart < len(starts) else rng.uniform(0, 1, 10)
                restart += 1
                value = objective(x)
                step = .2
                while step >= .00625:
                    improved = False
                    for j in rng.permutation(len(x)):
                        next_x, next_score = x.copy(), value
                        for sign in [-1, 1]:
                            candidate = x.copy()
                            candidate[j] = np.clip(candidate[j]+sign*step, 0, 1)
                            if np.array_equal(candidate, x):
                                continue
                            score = objective(candidate)
                            if score > next_score+1e-10:
                                next_x, next_score = candidate, score
                        if next_score > value+1e-10:
                            x, value, improved = next_x, next_score, True
                    if not improved:
                        step /= 2
        except Done:
            pass
    seconds = time.perf_counter()-start
    if best_x is None:
        raise ProjectionFailure("失敗：直線状または点に潰れない散布図が見つかりませんでした。")
    return {"initial_parameters": initial, "parameters": decode(best_x), "trials": len(history),
            "optimization_seconds": seconds, "history": history, "seed": seed, "stop_reason": "評価回数上限"}


def separation_evidence(X, Z, M, y, intent):
    if intent.pattern == "target":
        pairs = [(y == intent.targets[0], y != intent.targets[0], LABELS[intent.targets[0]], "その他")]
    else:
        ids = intent.targets if intent.pattern == "pair" else list(range(3))
        pairs = [(y == a, y == b, LABELS[a], LABELS[b]) for a, b in itertools.combinations(ids, 2)]
    evidence = []
    for a, b, name_a, name_b in pairs:
        delta = Z[a].mean(0)-Z[b].mean(0)
        distance = float(np.linalg.norm(delta))
        d = delta/distance if distance > 1e-12 else np.zeros(2)
        direction = "重心が一致" if distance <= 1e-12 else ("横方向" if abs(d[0]) >= .924 else "縦方向" if abs(d[1]) >= .924 else "斜め方向")
        differences = X[a].mean(0)-X[b].mean(0)
        directional_loading = M@d
        contributions = differences*directional_loading
        evidence.append({"group_a": name_a, "group_b": name_b, "direction": direction,
            "centroid_delta": delta.tolist(), "unit_direction": d.tolist(), "distance": distance,
            "features": [{"name": f, "mean_difference_scaled": float(differences[j]),
                          "directional_loading": float(directional_loading[j]),
                          "separation_contribution": float(contributions[j])} for j, f in enumerate(FEATURES)]})
    return evidence


def png(fig):
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=130, bbox_inches="tight")
    return "data:image/png;base64,"+base64.b64encode(stream.getvalue()).decode()


def plots(initial_Z, Z, y, M, history, objective="silhouette"):
    fig = Figure(figsize=(11, 4.6))
    limits = np.vstack([initial_Z, Z])
    extent = [limits[:,0].min()-.3, limits[:,0].max()+.3, limits[:,1].min()-.3, limits[:,1].max()+.3]
    for ax, points, title in zip(fig.subplots(1, 2), [initial_Z, Z], ["Semantic initialization", "After coordinate search"]):
        for k in range(3):
            p = points[y == k]
            mu = p.mean(0)
            ax.scatter(*p.T, s=20, c=COLORS[k], alpha=.75, label=LABELS[k])
            ax.scatter(*mu, s=100, c=COLORS[k], marker="X", edgecolors="black")
        ax.axis(extent)
        ax.set_aspect("equal", adjustable="box")
        ax.set(xlabel="ULCA 1", ylabel="ULCA 2", title=title)
        ax.legend(fontsize=8)
    scatter = png(fig)
    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    if NAME == "mnist069":
        fig = Figure(figsize=(8, 4))
        axes = fig.subplots(1, 2)
        limit = max(float(np.abs(M).max()), 1e-12)
        for axis, plot in enumerate(axes):
            im = plot.imshow(M[:, axis].reshape(28, 28), cmap="coolwarm", vmin=-limit, vmax=limit)
            plot.set_title(f"Pixel loadings: ULCA {axis+1}")
            plot.set(xlabel="Column (0-based)", ylabel="Row (0-based)")
        fig.colorbar(im, ax=list(axes), shrink=.7)
    else:
        indices = np.arange(len(FEATURES))
        ax.barh(indices-.18, M[:,0], height=.35, label="ULCA 1", color="#2456a6")
        ax.barh(indices+.18, M[:,1], height=.35, label="ULCA 2", color="#de8c31")
        ax.set_yticks(indices, FEATURES)
        ax.invert_yaxis()
        ax.axvline(0, color="gray", linewidth=.7)
        ax.set(xlabel="Projection coefficient", title="Loadings · original variable order")
        ax.legend()
    loadings = png(fig)
    fig = Figure(figsize=(10, 3))
    ax = fig.subplots()
    ax.plot([h["trial"] for h in history], [h["best_score"] for h in history], color="#2456a6")
    ax.set(xlabel="ULCA evaluations", ylabel="Best variance ratio" if objective == "diversity" else "Best silhouette", title="Coordinate search")
    ax.grid(alpha=.2)
    return {"scatter": scatter, "loadings": loadings, "convergence": png(fig)}


def cluster_statistics(coordinates, labels):
    """Describe the actual point cloud used by both API and ellipse rendering."""
    return [
        {"class": k, "count": int(np.sum(labels == k)),
         "centroid": coordinates[labels == k].mean(0).tolist(),
         "variance": coordinates[labels == k].var(0).tolist(),
         "covariance": np.cov(coordinates[labels == k], rowvar=False, ddof=0).tolist()}
        for k in range(len(LABELS))
    ]


@threadpool_limits.wrap(limits=1)
def run_analysis(intent, budget=200, seed=17, *, optimize_parameters=True):
    started = time.perf_counter()
    raw = load_dataset()
    X = StandardScaler().fit_transform(raw.data)
    y = raw.target
    if optimize_parameters:
        result = optimize(X, y, intent, budget, seed)
    else:
        parameters = initial_parameters(intent)
        result = {"initial_parameters": parameters, "parameters": parameters,
                  "trials": 0, "optimization_seconds": 0., "history": [],
                  "seed": seed, "stop_reason": "初期表示（最適化なし）"}
    initial_model = EVDULCA()
    initial_Z = initial_model.fit_transform(X, y, **result["initial_parameters"])
    model = EVDULCA()
    Z = model.fit_transform(X, y, **result["parameters"])
    result.update({"intent": {"pattern": intent.pattern, "targets": intent.targets,
                             "name": PATTERNS[intent.pattern], "comparison": intent.description()},
        "dataset": f"{DISPLAY} ({len(y)} samples, {len(FEATURES)} features)", "variables": FEATURES, "colors": COLORS, "class_names": LABELS, "dataset_id": NAME,
        "initial_metrics": evaluate(initial_Z, y, intent), "metrics": evaluate(Z, y, intent),
        "coordinates": Z.tolist(), "initial_coordinates": initial_Z.tolist(), "labels": y.tolist(),
        "loadings": model.M.tolist(),
        "clusters": cluster_statistics(Z, y),
        "comparisons": [] if intent.pattern == "diversity" else separation_evidence(X, Z, model.M, y, intent),
        "images": plots(initial_Z, Z, y, model.M, result["history"], intent.pattern)})
    if intent.pattern == "diversity":
        result["metrics"] = evaluate(Z, y, intent, result["initial_metrics"]["target_variance"])
    result["recommended_variables"] = loading_recommendations(result)
    result["calculation_seconds"] = time.perf_counter()-started
    return result


def align_projection(reference, coordinates, model):
    if reference is None:
        return coordinates
    u, _, vh = np.linalg.svd(reference.T @ coordinates, full_matrices=False)
    rotation = vh.T @ u.T
    model.M = model.M @ rotation
    return coordinates @ rotation


@threadpool_limits.wrap(limits=1)
def update_projection(result, parameters):
    raw = load_dataset()
    X = StandardScaler().fit_transform(raw.data)
    y = raw.target
    intent = Intent(result["intent"]["pattern"], result["intent"]["targets"]).validate()
    model = EVDULCA()
    coordinates = model.fit_transform(X, y, **parameters)
    validate_projection(coordinates)
    coordinates = align_projection(np.asarray(result["coordinates"]), coordinates, model)
    updated = dict(result)
    updated.update(variables=FEATURES, parameters=parameters, coordinates=coordinates.tolist(),
        loadings=model.M.tolist(), metrics=evaluate(coordinates, y, intent),
        clusters=cluster_statistics(coordinates, y),
        comparisons=[] if intent.pattern == "diversity" else separation_evidence(X, coordinates, model.M, y, intent),
        images=plots(np.asarray(result["initial_coordinates"]), coordinates, y, model.M,
                     result["history"], intent.pattern),
        manual_adjustment=True)
    if intent.pattern == "diversity":
        baseline = evaluate(np.asarray(result["initial_coordinates"]), y, intent)["target_variance"]
        updated["metrics"] = evaluate(coordinates, y, intent, baseline)
    updated["recommended_variables"] = loading_recommendations(updated)
    updated["explanation"] = explain(updated)
    return updated


def inverse_adjust(result, target_geometry, updated_label, interaction, max_iterations=50):
    current = result["parameters"]
    initial = encode(current)
    labels = np.asarray(result["labels"])
    reference = np.asarray(result["coordinates"])
    geometry = sorted(target_geometry, key=lambda item: item["label"])
    ideal_areas = np.array([max(float(item["area"]), 1e-8) for item in geometry])
    ideal_centers = np.array([item["center"] for item in geometry], dtype=float)
    ideal_distances = pdist(ideal_centers)
    area_weight, distance_weight = ((.2, .8) if interaction == "move" else (.8, .2))
    X = StandardScaler().fit_transform(load_dataset().data)
    best_values = initial.copy()
    best_score = np.inf

    def objective(values):
        nonlocal best_values, best_score
        parameters = decode(values)
        model = EVDULCA()
        coordinates = model.fit_transform(X, labels, **parameters)
        try:
            validate_projection(coordinates)
        except ProjectionFailure:
            return 1e6
        coordinates = align_projection(reference, coordinates, model)
        centers = np.array([coordinates[labels == k].mean(axis=0) for k in range(3)])
        areas = np.array([
            np.sqrt(max(0, np.linalg.det(np.cov(
                coordinates[labels == k], rowvar=False, ddof=0))))
            for k in range(3)
        ])
        ideal_ratios = ideal_areas[updated_label] / ideal_areas
        actual_ratios = areas[updated_label] / np.maximum(areas, 1e-12)
        area_loss = np.mean(np.minimum(
            1, np.abs((ideal_ratios - actual_ratios) / ideal_ratios)))
        actual_distances = pdist(centers)
        distance_scale = np.linalg.norm(ideal_distances)
        distance_loss = (np.linalg.norm(actual_distances - ideal_distances) /
                         distance_scale if distance_scale > 1e-12 else
                         1.0)
        score = float(area_weight * area_loss + distance_weight * distance_loss)
        if np.isfinite(score) and score < best_score:
            best_values = np.asarray(values).copy()
            best_score = score
        return score

    bounds = [(0, 1)] * 9 + [(0, 1)]
    with threadpool_limits(limits=1):
        initial_score = objective(initial)
        optimized = minimize(objective, initial, method="SLSQP", bounds=bounds,
                             options={"maxiter": max_iterations, "ftol": 1e-5})
    updated = update_projection(result, decode(best_values))
    updated["inverse_adjustment"] = {
        "success": bool(best_score < initial_score - 1e-8),
        "converged": bool(optimized.success),
        "message": str(optimized.message),
        "iterations": int(optimized.nit),
        "objective_before": float(initial_score),
        "objective": float(best_score),
    }
    return updated


def loading_recommendations(result):
    """Rank raw axis coefficients, independent of LLM text and centroid products."""
    axes = set()
    diversity = result["intent"].get("pattern") == "diversity"
    if diversity:
        axes.update([0, 1])
    else:
        for item in result.get("comparisons", []):
            if item.get("direction", "斜め方向") == "横方向":
                axes.add(0)
            elif item.get("direction", "斜め方向") == "縦方向":
                axes.add(1)
            elif item.get("direction", "斜め方向") != "重心が一致":
                axes.update([0, 1])
    count = 2 if diversity else 3
    ranked = []
    for axis in sorted(axes):
        indices = sorted(range(len(result["variables"])),
                         key=lambda j: (-abs(result["loadings"][j][axis]), j))[:count]
        ranked.append({"axis": "XY"[axis], "features": [
            {"name": result["variables"][j], "loading": float(result["loadings"][j][axis])}
            for j in indices]})
    return ranked


def variable_explanation(result):
    lines = ["2. 寄与した変数："]
    for item in loading_recommendations(result):
        names = "、".join(f'{f["name"]} ({f["loading"]:.2f})' for f in item["features"])
        lines.append(f'{item["axis"]}軸：{names}。')
    if len(lines) == 1:
        lines.append("重心が一致しているため、分離方向の変数は提示しません。")
    return "\n".join(lines)


def projection_situation(result):
    if result["intent"].get("pattern") == "diversity":
        target_id = result["intent"]["targets"][0]
        target = LABELS[target_id]
        points = np.asarray(result["coordinates"])[np.asarray(result["labels"]) == target_id]
        variance = points.var(axis=0)
        direction = "X軸方向" if variance[0] > variance[1]*1.5 else "Y軸方向" if variance[1] > variance[0]*1.5 else "X軸・Y軸方向"
        return f"{target}の内部変動は、主に{direction}に広がっています。"
    lines = []
    for item in result.get("comparisons", []):
        if item["direction"] == "重心が一致":
            lines.append(f'{item["group_a"]}と{item["group_b"]}の重心は一致しています。')
        else:
            lines.append(f'{item["group_a"]}と{item["group_b"]}の重心は{item["direction"]}に分かれています。')
    return "\n".join(lines)


def explain(result, situation=None):
    # Only the spatial description can come from the LLM. Feature names,
    # coefficients and their ordering always come from the full numeric matrix.
    description = situation or projection_situation(result)
    return f"1. 散布図の状況：{description}\n\n{variable_explanation(result)}"


def explain_with_llm(result, text):
    # Discard a legacy variable section if a model ignores the short-text contract.
    text = re.split(r"(?:2[.．、)）]\s*|(?:\*\*)?)寄与した変数", text, maxsplit=1)[0]
    text = text.rstrip(" \n\r-#*")
    text = re.sub(r"^[\s#>*-]*(?:1[.．、)）]\s*)?(?:\*\*)?散布図の状況(?:\*\*)?\s*[:：]\s*", "", text).strip()
    # Do not let invented variable prose or coefficients leak into the situation.
    if not text or any(re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.I)
                       for name in result["variables"]):
        text = projection_situation(result)
    return explain(result, text[:240])
