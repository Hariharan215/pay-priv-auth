"""
Server package for Pay‑PrivAuth.

Contains the risk engine simulation used to combine behavioural authentication
scores with other risk signals to decide whether a transaction should be
approved, challenged or denied.  In a real deployment this module would be
integrated into an issuer's access control server (ACS) and communicate
with client devices via EMV 3‑D Secure.
"""

from .risk_engine import RiskEngine

__all__ = ["RiskEngine"]