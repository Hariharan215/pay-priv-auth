.PHONY: setup features train run_exp api

setup:
pip install -r requirements.txt

features:
python -m pay_priv_auth.build_dataset --root data/raw/HMOG --out data/processed/hmog_features.parquet

train:
python pay_priv_auth/train.py

run_exp:
python pay_priv_auth/run_experiment.py

api:
uvicorn pay_priv_auth.server.app:app --port 8088 --reload
