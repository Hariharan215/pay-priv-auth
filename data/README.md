# Data directory

This project currently supports **HMOG** and **BehavePassDB** datasets.
Place the raw archives under:

- `data/raw/HMOG/`
- `data/raw/BehavePassDB/`

After running the loaders each session directory should contain normalised
files:

- `touch.csv` with columns `t,x,y,pressure,size,event_type`
- `keys.csv` with columns `t,event_type,key_code`
- `imu.csv` with columns `t,ax,ay,az,gx,gy,gz`
- `meta.json` describing `user_id`, `session_id`, `phase`

Plaintext content is never stored; only timings and sensor values.

Build feature datasets with:

```bash
python -m pay_priv_auth.build_dataset --dataset hmog --root data/raw/HMOG --out data/processed/hmog_features.parquet
python -m pay_priv_auth.build_dataset --dataset behavepass --root data/raw/BehavePassDB --out data/processed/behavepass_features.parquet
```
