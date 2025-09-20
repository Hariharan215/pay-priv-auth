# Pay-PrivAuth

Minimal framework for privacy-preserving behavioural authentication in
card-not-present payments. The project demonstrates feature extraction
from raw interaction data, per-user logistic models, differential
privacy primitives and a toy risk engine API.

This project uses the [BBMAS dataset](https://ieee-dataport.org/open-access/su-ais-bb-mas-syracuse-university-and-assured-information-security-behavioral) (Syracuse University and Assured Information Security Behavioral Dataset) for behavioral authentication. We focus specifically on:
- Keyboard and mouse interaction data for desktop environments
- Cleaning and preprocessing to exclude mobile-related data (gyro, accelerometer, etc.)
- Labeling and clustering of anomalous entries
- Train/test split using a simple SVM model
- Real-time streaming predictions for normal/anomalous behavior
- Implementation of federated learning and differential privacy for user data anonymization

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
- Federated learning implementation for distributed model training
- Differential privacy applied to anonymize user behavioral data

## Next steps
- Android capture SDK.
- Integration with live payment risk engines.
- Enhanced real-time anomaly detection pipeline
- Expansion of federated learning capabilities
