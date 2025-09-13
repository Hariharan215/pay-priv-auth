.PHONY: setup features-hmog features-behavepass train-hmog train-behavepass api

setup:
pip install -r requirements.txt

features-hmog:
python -m pay_priv_auth.build_dataset --dataset hmog --root data/raw/HMOG --out data/processed/hmog_features.parquet

features-behavepass:
python -m pay_priv_auth.build_dataset --dataset behavepass --root data/raw/BehavePassDB --out data/processed/behavepass_features.parquet

train-hmog:
python pay_priv_auth/train.py --dataset_parquet data/processed/hmog_features.parquet

train-behavepass:
python pay_priv_auth/train.py --dataset_parquet data/processed/behavepass_features.parquet

api:
uvicorn pay_priv_auth.server.app:app --port 8088 --reload
