from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import operator

# ── State Schema ──────────────────────────────────────────
class MortgageState(TypedDict):
    lead_id: str
    applicant_name: str
    annual_income: float
    loan_amount: float
    credit_score: int
    debt_to_income: float
    guidelines_retrieved: list[str]
    pre_approval_decision: str
    rationale: str
    bias_flags: list[str]
    audit_log: Annotated[list[str], operator.add]

# ── Node 1: Parse Lead ─────────────────────────────────────
def parse_lead(state: MortgageState) -> MortgageState:
    log = f"[PARSE] Lead {state['lead_id']} ingested for {state['applicant_name']}"
    print(log)
    return {"audit_log": [log]}

# ── Node 2: Retrieve Compliance Guidelines (stub) ──────────
def retrieve_guidelines(state: MortgageState) -> MortgageState:
    guidelines = [
        "Fannie Mae B3-3.1: DTI generally not to exceed 45%.",
        "Freddie Mac 5306.1: Minimum credit score 620 for conventional loans.",
    ]
    log = f"[RAG] Retrieved {len(guidelines)} guideline chunks."
    print(log)
    return {"guidelines_retrieved": guidelines, "audit_log": [log]}

# ── Node 3: Pre-Approval Decision ─────────────────────────
def pre_approval_agent​​​​​​​​​​​​​​​​
