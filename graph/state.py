# app/graph/state.py
from typing import TypedDict, Annotated, List, Dict, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """Main state for our Due Diligence Agent"""
    messages: Annotated[List[BaseMessage], add_messages]
    session_id: str
    company_name: str
    
    # Due Diligence Specific Fields
    plan: Optional[List[str]]                    # Due diligence steps
    retrieved_context: Optional[str]
    analysis_per_section: Optional[Dict[str, str]]  # Team, Market, Traction, etc.
    risks: Optional[List[str]]
    opportunities: Optional[List[str]]
    overall_score: Optional[float]
    final_memo: Optional[str]
    sources: List[Dict]
    next: str