import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend import dataset
from backend.app import app
from backend.engine import classify, Intent, run_analysis, update_projection, inverse_adjust

pytestmark = pytest.mark.skipif(dataset.NAME != 'mnist069', reason='MNIST configuration only')

def test_mnist_dataset_and_digit_mapping():
    raw=dataset.load_dataset()
    assert raw.data.shape==(1500,784)
    assert np.array_equal(np.bincount(raw.target),[500,500,500])
    assert raw.data.min()>=0 and raw.data.max()<=255
    assert classify('数字6と9を比較して').targets==[1,2]
    assert classify('Digit 6の特徴を教えて').targets==[1]
    with pytest.raises(ValueError): classify('数字1を分離して')

def test_mnist_initial_manual_inverse():
    c=TestClient(app)
    d=c.get('/api/dataset').json()
    assert d['class_names']==['Digit 0','Digit 6','Digit 9']
    r=c.get('/api/initial').json()
    assert len(r['coordinates'])==1500 and len(r['loadings'])==784
    p=dict(r['parameters'],alpha=5.)
    updated=update_projection(r,p)
    assert np.isfinite(updated['coordinates']).all()
    geometry=[dict(label=k,center=s['centroid'],area=max(1e-8,np.sqrt(max(0,np.linalg.det(s['covariance']))))) for k,s in enumerate(updated['clusters'])]
    inverse=inverse_adjust(updated,geometry,0,'move',10)
    assert np.isfinite(inverse['coordinates']).all()
    assert inverse['inverse_adjustment']['objective']<=inverse['inverse_adjustment']['objective_before']+1e-8


def test_mnist_collapsed_manual_change_is_rejected():
    c = TestClient(app)
    r = c.get('/api/initial').json()
    original = np.asarray(r['coordinates']).copy()
    response = c.post('/api/update', json={
        'pattern': 'overall', 'targets': [],
        'parameters': dict(w_tg=[0., 0., 0.], w_bg=[1., 1., 1.],
                           w_bw=[1., 0., 0.], alpha=10.),
        'reference_coordinates': r['coordinates'],
        'initial_coordinates': r['initial_coordinates'], 'history': []})
    assert response.status_code == 422
    assert response.json()['status'] == 'failed'
    assert '直線状' in response.json()['detail']
    assert np.array_equal(r['coordinates'], original)
