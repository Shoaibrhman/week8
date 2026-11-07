"""Latency benchmark against a RUNNING server: python bench.py"""
import os
import platform
import time

import httpx
import numpy as np

URL, N = 'http://127.0.0.1:8000/predict', 50
lat = []
with httpx.Client(timeout=30) as client, open('sample.jpeg', 'rb') as f:
    data = f.read()
    for _ in range(3):                                  # warm-up, not counted
        client.post(URL, files={'file': ('x.jpeg', data, 'image/jpeg')})
    for _ in range(N):
        t0 = time.perf_counter()
        r = client.post(URL, files={'file': ('x.jpeg', data, 'image/jpeg')})
        r.raise_for_status()
        lat.append((time.perf_counter() - t0) * 1000)
lat = np.array(lat)
print(f'Hardware: {platform.processor() or platform.machine()}, '
      f'{os.cpu_count()} logical cores, {platform.system()} (add your RAM)')
print(f'n={N} p50={np.percentile(lat, 50):.0f} ms '
      f'p95={np.percentile(lat, 95):.0f} ms max={lat.max():.0f} ms')
print(f'Throughput with 1 worker ~ {1000 / lat.mean():.1f} requests/s')
print(f'Little\'s law, 1 worker at p50: '
      f'{86400 * 1000 / np.percentile(lat, 50):,.0f} requests/day at 100% busy')
