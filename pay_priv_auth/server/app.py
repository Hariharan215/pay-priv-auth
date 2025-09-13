"""FastAPI service exposing the risk engine."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .risk_engine import RiskEngine

app = FastAPI()
engine = RiskEngine()


class FrictionlessReq(BaseModel):
    tx_id: str
    device_id: str
    model_ver: str
    score: float
    eps_budget: float
    rba_low_risk: bool


class StepUpReq(BaseModel):
    tx_id: str
    score_otp: float
    otp_valid: bool


@app.post("/frictionless")
async def frictionless(req: FrictionlessReq):
    decision = engine.decide_frictionless(req.score, req.rba_low_risk)
    return {"decision": decision}


@app.post("/stepup")
async def stepup(req: StepUpReq):
    decision = engine.decide_stepup(req.score_otp, req.otp_valid)
    return {"decision": decision}
