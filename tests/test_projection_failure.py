import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend.app import app
from backend import engine


def test_rejects_rotated_line_and_point_but_accepts_plane():
    t = np.linspace(-2, 2, 40)
    for points in [np.column_stack([t, 3*t]), np.zeros((40, 2)),
                   1e-12*np.column_stack([t, t*t]),
                   np.column_stack([t, 1e-4*t*t])]:
        with pytest.raises(engine.ProjectionFailure):
            engine.validate_projection(points)
    engine.validate_projection(np.column_stack([t, t*t]))


def test_search_excludes_failed_candidates(monkeypatch):
    calls = 0
    t = np.linspace(-2, 2, 12)
    class Model:
        def fit_transform(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            return np.column_stack([t, t if calls % 2 else t*t])
    monkeypatch.setattr(engine, 'EVDULCA', Model)
    result = engine.optimize(np.zeros((12, 2)), np.repeat([0, 1, 2], 4),
                             engine.Intent('overall', []), budget=10)
    assert result['trials'] == 10
    assert sum(h.get('valid') is False for h in result['history']) == 5
    assert result['history'][-1]['best_score'] > -1


def test_failed_projection_api_message(monkeypatch):
    def fail(*args, **kwargs):
        raise engine.ProjectionFailure('失敗：散布図が直線状または点に潰れました。')
    monkeypatch.setattr('backend.app.run_analysis', fail)
    response = TestClient(app).get('/api/initial')
    assert response.status_code == 422
    assert response.json()['status'] == 'failed'
    assert '直線状' in response.json()['detail']
