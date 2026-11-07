## Week 8: Serving the model as an API

FastAPI service around the Week 6 crop-disease ResNet-18 (38 PlantVillage classes) with Week 7 note handling (PII masking, urgent terms).

### Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | /health | status, model_loaded, classes, positive (diseased) classes |
| POST | /predict | leaf image -> label, 38 probabilities, latency_ms |
| POST | /analyze | leaf image + farmer note -> masked note, urgent terms, REFER/ROUTINE decision with reasons |

### How to run
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn python-multipart pillow pytest httpx numpy
uvicorn app:app --reload          # then open http://127.0.0.1:8000/docs
pytest -q test_api.py
python bench.py                   # with the server running
```
Configuration (environment variables): `MODEL_PATH`, `POSITIVE` (comma-separated diseased classes; default = every class that is not healthy/background), `THRESHOLD` (default 0.5), `API_KEY` (optional).

### Test results
```
........                                                          [100%]
8 passed, 1 warning in 7.38s
```
The warning is a Starlette deprecation notice about the httpx test client; it does not affect results.

### Latency benchmark
Hardware: Intel Core i7-1355U (12 logical cores), 16 GB RAM, Windows, CPU-only PyTorch, 1 uvicorn worker.

| n | p50 | p95 | max | throughput (1 worker) |
|---|---|---|---|---|
| 50 sequential requests | 119 ms | 123 ms | 124 ms | about 8.5 req/s |

The tail is tight (123 vs 119 ms) because requests were sequential; under concurrent load it would stretch.

**Little's law:** one worker handles about 1000 / 119 = 8.4 requests per second, which is 86,400 x 8.4, about 727,000 requests per day at 100% busy and roughly 360,000 per day at a realistic 50% utilisation. That is far above what a student product needs.

### Bad inputs tried
| Input | Response |
|---|---|
| text/plain file (app.py) | 415 `Upload a JPEG or PNG image` |
| 6 MB file | 413 `Image larger than 5 MB` |
| not-an-image bytes (test) | 400 `File is not a readable image` |
| missing file (test) | 422 |
| **Extra input beyond the tests: 0-byte file sent as image/jpeg** | 400 `File is not a readable image` |

None returned a stack trace. PII check: a note containing a phone number came back as `Call [PHONE]`, and the test asserts that no digit sequence of the phone number or CNIC appears anywhere in the response.

### Example /analyze output
Note: `Dark spots spreading on leaves. Call 0300-1234567`
```
"note_masked": "Dark spots spreading on leaves. Call [PHONE]",
"urgent_terms": ["spot", "spreading"],
"decision": "REFER",
"reasons": ["image: P(diseased) = 1.00 (top: Apple___Apple_scab)", "note mentions: spot, spreading"]
```

### Changes from the lab template
- Multi-class "positive": P(diseased) is the sum of all non-healthy class probabilities, because a single-class threshold does not work with 38 classes.
- Urgent words are matched on word starts (so "root" does not trigger "rot").
- Decompression-bomb images return 400 instead of a 500.
- Probability-sum test tolerance relaxed to 5e-3 because 38 values rounded to 4 decimals can drift past 1e-3.

### Check your understanding
1. **PDF to /predict:** 415 Unsupported Media Type. It is the client's mistake and fixable; a 500 would mean a bug in our code, and clients rely on the 4xx/5xx difference.
2. **Why lifespan:** loading weights takes seconds while a request should take milliseconds. Loading once at startup avoids that cost per request, keeps one copy in memory, and builds the transform from the checkpoint so preprocessing cannot drift from training.
3. **p50 = 40 ms, p95 = 400 ms:** typical requests are fast but 1 in 20 is ten times slower. Check event-loop blocking, queueing under concurrency, oversized inputs, CPU contention and cold starts; look at max and the whole distribution.
4. **500,000 requests/day, peak 5x, 25 req/s per worker:** average 5.8 req/s, peak about 29 req/s, 29 / 25 = 1.16, so 2 workers (a third for headroom).
5. **Inference inside async def:** an async server has one event loop; CPU-heavy code that never yields freezes it, so every other request, even /health, waits. Use run_in_threadpool or a plain def endpoint.
6. **4-person team:** 4 x 3 / 2 = 6 communication channels.

### Reflection
Turning my Week 6 crop-disease model into an API showed me that a good notebook score is not the same as a working product: the service also has to reject bad files (415, 413, 400), hide personal data like phone numbers, and be tested automatically, and all 8 of my tests now pass. My benchmark gave p50 119 ms and p95 123 ms on an i7-1355U CPU, which is slower than the 45 ms in the lab example, so I learned to report my own hardware instead of copying someone else's numbers. The biggest lesson from earlier weeks still applies: my border-only test scored 0.238 against 0.026 by chance, so the model may use the background and not only the disease, which is why the API says a qualified agronomist makes the final decision.
