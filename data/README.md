# Data directory

Place raw behavioural datasets under `data/raw/<DATASET>/<user>/<session>/`.
Each session directory should contain:

- `touch.csv` with columns `t,x,y,pressure,size,event_type`
- `keys.csv` with columns `t,event_type,key_code`
- `imu.csv` with columns `t,ax,ay,az,gx,gy,gz`
- `meta.json` describing `user_id`, `session_id`, `phase`

No plaintext content is stored; only timings and coordinates.
Run feature extraction with:

```
python -m pay_priv_auth.build_dataset --root data/raw/<DATASET> --out data/processed/<dataset>_features.parquet
```
