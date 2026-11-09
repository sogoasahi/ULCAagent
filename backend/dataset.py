"""Reversible dataset adapter. Select wine or mnist069 in dataset.json."""
import gzip
import json
import os
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.datasets import get_data_home, load_wine

ROOT = Path(__file__).resolve().parent
NAME = os.environ.get('ULCA_DATASET') or json.loads((ROOT/'dataset.json').read_text())['name']
if NAME not in ('wine', 'mnist069'):
    raise ValueError('ULCA_DATASET must be wine or mnist069')
WINE_FEATURES = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
                 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315', 'Proline']
FEATURES = WINE_FEATURES if NAME == 'wine' else [f'Pixel({r+1},{c+1})' for r in range(28) for c in range(28)]
LABELS = ['Barolo', 'Grignolino', 'Barbera'] if NAME == 'wine' else ['Digit 0', 'Digit 6', 'Digit 9']
DISPLAY = 'Wine' if NAME == 'wine' else 'MNIST 0 / 6 / 9'
ALIASES = ([('barolo', 'baroro', 'baloro', 'バローロ', '赤'), ('grignolino', 'グリニョリーノ', '緑'),
            ('barbera', 'バルベーラ', '紫')] if NAME == 'wine' else [('赤',), ('緑',), ('紫',)])

@lru_cache(maxsize=1)
def load_dataset():
    if NAME == 'wine':
        return load_wine()
    # Read only the existing cache; never download data implicitly.
    source = Path(get_data_home())/'openml/openml.org/data/v1/download/52667/mnist_784.arff.gz'
    if not source.exists():
        raise FileNotFoundError(f'MNIST local cache not found: {source}')
    samples = {0: [], 6: [], 9: []}
    reading = False
    with gzip.open(source, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if not reading:
                reading = line.strip().lower() == '@data'
                continue
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            fields = line.split(',')
            digit = int(fields[-1].strip("'\""))
            if digit in samples and len(samples[digit]) < 500:
                pixels = np.asarray(fields[:-1], dtype=np.float64)
                if len(pixels) != 784 or not np.isfinite(pixels).all():
                    raise ValueError('Invalid MNIST sample')
                samples[digit].append(pixels)
            if all(len(v) == 500 for v in samples.values()):
                break
    if any(len(v) != 500 for v in samples.values()):
        raise ValueError('Expected 500 samples per digit')
    return SimpleNamespace(data=np.vstack(list(samples.values())), target=np.repeat([0, 1, 2], 500))

def dataset_context():
    if NAME == 'wine':
        return ''
    return ('現在はMNIST手書き数字0・6・9です。内部IDは0=Digit 0/赤、1=Digit 6/緑、2=Digit 9/紫。'
            'ユーザーの数字6は内部ID1、数字9は内部ID2です。Wineの品種名・例は適用しません。'
            '変数は28×28画像の画素でPixel(行,列)、左上が(1,1)です。元の値は0〜255の明るさです。'
            '変数名はそのまま使用し、画素をワインの成分として説明しないでください。')
