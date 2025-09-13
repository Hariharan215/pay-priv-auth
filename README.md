# Pay-PrivAuth

Minimal framework for privacy-preserving behavioural authentication in
card-not-present payments. The project demonstrates feature extraction
from raw interaction data, per-user logistic models, differential
privacy primitives and a toy risk engine API.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make features       # builds features from data/raw if present
python pay_priv_auth/train.py   # trains baseline with synthetic fallback
uvicorn pay_priv_auth.server.app:app --port 8088 --reload
```

## Notes on privacy
- Only timing/coordinates are processed, no raw text.
- Embeddings use random projection, quantisation and small noise.
- Future work: on-device processing, DP budgets, real 3-DS wiring.

## Next steps
- Android capture SDK.
- Integration with live payment risk engines.
