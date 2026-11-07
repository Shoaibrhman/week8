"""Run with: pytest -q test_api.py"""
import io

import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient
from PIL import Image

import app as api

SAMPLE = 'sample.jpeg'


def post_image(client, path=SAMPLE, ctype='image/jpeg', **kw):
    with open(path, 'rb') as f:
        return client.post('/predict', files={'file': ('x', f, ctype)}, **kw)


def test_health():
    with TestClient(api.app) as c:
        r = c.get('/health')
        assert r.status_code == 200 and r.json()['model_loaded']


def test_predict_schema_and_probabilities():
    with TestClient(api.app) as c:
        r = post_image(c)
        assert r.status_code == 200
        probs = r.json()['probabilities']
        # 38 classes rounded to 4 decimals can drift a little more than 1e-3
        assert abs(sum(probs.values()) - 1) < 5e-3


def test_parity_with_direct_model():
    with TestClient(api.app) as c:
        served = post_image(c).json()['probabilities']
        ck, tf = api.state['ck'], api.state['tf']
        img = Image.open(SAMPLE).convert('L' if ck['grayscale'] else 'RGB')
        with torch.inference_mode():
            p = F.softmax(api.state['model'](tf(img).unsqueeze(0)), 1)[0]
        direct = dict(zip(ck['classes'], p.tolist()))
        for k in direct:
            assert abs(served[k] - direct[k]) < 1e-3   # same numbers


def test_rejects_wrong_type():
    with TestClient(api.app) as c:
        r = c.post('/predict', files={'file': ('a.txt', b'hello',
                                               'text/plain')})
        assert r.status_code == 415


def test_rejects_too_large():
    big = io.BytesIO(b'0' * (api.MAX_BYTES + 1))
    with TestClient(api.app) as c:
        r = c.post('/predict', files={'file': ('b.jpg', big, 'image/jpeg')})
        assert r.status_code == 413


def test_rejects_corrupt_image():
    with TestClient(api.app) as c:
        r = c.post('/predict', files={'file': ('c.jpg', b'not an image',
                                               'image/jpeg')})
        assert r.status_code == 400


def test_missing_file_is_422():
    with TestClient(api.app) as c:
        assert c.post('/predict').status_code == 422


def test_analyze_masks_pii():
    note = ('Leaves have dark spots spreading fast. '
            'Farmer 0300-1234567, CNIC 35202-1234567-1')
    with TestClient(api.app) as c, open(SAMPLE, 'rb') as f:
        r = c.post('/analyze', files={'file': ('x', f, 'image/jpeg')},
                   data={'note': note})
        body = r.json()
        assert r.status_code == 200
        assert '1234567' not in str(body)              # no leak anywhere
        assert '35202' not in str(body)
        assert body['decision'] == 'REFER'
