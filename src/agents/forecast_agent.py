"""
src/agents/forecast_agent.py
─────────────────────────────
LangGraph-based agent that:
1. Takes TrendReport objects from the trend engine
2. Calls Claude claude-sonnet-4-20250514 to synthesize trend narratives
3. Generates structured StyleCard outputs
4. Produces a ForecastReport with actionable insights

Graph topology:
    [prepare_context] → [synthesize_trends] → [generate_style_cards]
                                 ↓
                        [rank_and_format] → END
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, TypedDict

# Lazy import Anthropic
def _get_anthropic():
    try:
        import anthropic
        return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    except ImportError:
        return None


# ── Output data models ────────────────────────────────────────────────────────

@dataclass
class StyleCard:
    """A generated style card for a single trend."""
    trend_name: str
    headline: str                        # 1-line punchy headline
    narrative: str                       # 2-3 paragraph trend story
    key_pieces: list[str]                # 5 essential items
    styling_tips: list[str]              # 3 how-to-wear tips
    color_story: str                     # color palette description
    target_demographic: str
    price_entry_point: str
    where_to_shop: list[str]
    forecast_horizon: str                # "6 months", "1 year", etc.
    velocity: str
    confidence: float                    # 0-1, how confident the LLM is


@dataclass
class ForecastReport:
    """Full forecast output — what the API returns."""
    generated_at: str
    analysis_window_days: int
    total_trends_analyzed: int
    rising_count: int
    declining_count: int
    executive_summary: str
    macro_direction: str                 # overall fashion macro trend
    style_cards: list[StyleCard]
    raw_trend_contexts: list[str] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("raw_trend_contexts", None)
        return d


# ── LangGraph State ───────────────────────────────────────────────────────────

class AgentState(TypedDict):
    trend_reports: list[Any]         # list[TrendReport]
    contexts: list[str]
    raw_synthesis: str
    style_cards: list[StyleCard]
    executive_summary: str
    macro_direction: str
    error: Optional[str]


# ── Individual node functions ─────────────────────────────────────────────────

def prepare_context(state: AgentState) -> AgentState:
    """Convert TrendReport objects to LLM-friendly text blocks."""
    contexts = []
    for i, report in enumerate(state["trend_reports"], 1):
        if hasattr(report, "to_llm_context"):
            ctx = f"[TREND {i}]\n{report.to_llm_context()}"
        else:
            ctx = f"[TREND {i}]\n{json.dumps(report, indent=2)}"
        contexts.append(ctx)
    return {**state, "contexts": contexts}


def synthesize_trends(state: AgentState) -> AgentState:
    """Call Claude to synthesize trend data into narrative insights."""
    client = _get_anthropic()
    if not client:
        return {**state, "error": "Anthropic client not available", "raw_synthesis": "{}"}

    context_block = "\n\n".join(state["contexts"])

    system_prompt = """You are TrendAI, a world-class fashion trend analyst and forecaster.
You combine the rigor of a quantitative analyst with the intuition of a seasoned fashion editor.
Your forecasts are used by buyers, designers, and content creators.

Always respond with valid JSON only — no markdown fences, no preamble."""

    user_prompt = f"""Analyze these {len(state['trend_reports'])} fashion trend clusters detected from social media, runway, and retail data over the past analysis window.

{context_block}

Return a JSON object with this exact structure:
{{
  "executive_summary": "2-3 sentence overview of the current fashion moment",
  "macro_direction": "one-sentence description of the overarching macro trend direction",
  "style_cards": [
    {{
      "trend_name": "exact name from input",
      "headline": "punchy 10-word headline for this trend",
      "narrative": "2-3 paragraph story of this trend: its origin, current expression, and trajectory",
      "key_pieces": ["piece 1", "piece 2", "piece 3", "piece 4", "piece 5"],
      "styling_tips": ["tip 1", "tip 2", "tip 3"],
      "color_story": "1 sentence describing the color palette and mood",
      "target_demographic": "primary audience (age range, lifestyle)",
      "price_entry_point": "e.g. '$50–$200 for high-street, $500+ for investment pieces'",
      "where_to_shop": ["brand/retailer 1", "brand/retailer 2", "brand/retailer 3"],
      "forecast_horizon": "e.g. 'Peak in 3–6 months, then gradual decline'",
      "confidence": 0.85
    }}
  ]
}}

Write for a sophisticated fashion-industry audience. Be specific, opinionated, and forward-looking.
Only include a style card for trends with velocity 'rising' or 'peak'. Skip 'declining' trends.
Limit to the 5 most compelling trends."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = message.content[0].text
        return {**state, "raw_synthesis": raw}
    except Exception as e:
        return {**state, "error": str(e), "raw_synthesis": "{}"}


def generate_style_cards(state: AgentState) -> AgentState:
    """Parse raw LLM JSON into structured StyleCard objects."""
    if state.get("error"):
        return state

    try:
        text = state["raw_synthesis"]
        # Strip accidental markdown fences
        text = re.sub(r"```json|```", "", text).strip()
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return {**state, "error": f"JSON parse error: {e}", "style_cards": []}

    cards = []
    for sc_data in data.get("style_cards", []):
        card = StyleCard(
            trend_name=sc_data.get("trend_name", "Unknown"),
            headline=sc_data.get("headline", ""),
            narrative=sc_data.get("narrative", ""),
            key_pieces=sc_data.get("key_pieces", []),
            styling_tips=sc_data.get("styling_tips", []),
            color_story=sc_data.get("color_story", ""),
            target_demographic=sc_data.get("target_demographic", ""),
            price_entry_point=sc_data.get("price_entry_point", ""),
            where_to_shop=sc_data.get("where_to_shop", []),
            forecast_horizon=sc_data.get("forecast_horizon", ""),
            velocity="rising",
            confidence=float(sc_data.get("confidence", 0.75)),
        )
        cards.append(card)

    return {
        **state,
        "style_cards": cards,
        "executive_summary": data.get("executive_summary", ""),
        "macro_direction": data.get("macro_direction", ""),
    }


def rank_and_format(state: AgentState) -> AgentState:
    """Sort style cards by confidence, attach velocity from trend reports."""
    reports_by_name = {}
    for r in state.get("trend_reports", []):
        name = getattr(r, "trend_name", None)
        if name:
            reports_by_name[name] = r

    cards = state.get("style_cards", [])
    for card in cards:
        report = reports_by_name.get(card.trend_name)
        if report:
            card.velocity = getattr(report, "velocity", card.velocity)

    cards.sort(key=lambda c: c.confidence, reverse=True)
    return {**state, "style_cards": cards}


# ── Agent builder ─────────────────────────────────────────────────────────────

def build_forecast_agent():
    """
    Build and return a compiled LangGraph agent.
    Falls back to a simple sequential runner if langgraph not installed.
    """
    try:
        from langgraph.graph import StateGraph, END

        graph = StateGraph(AgentState)
        graph.add_node("prepare_context", prepare_context)
        graph.add_node("synthesize_trends", synthesize_trends)
        graph.add_node("generate_style_cards", generate_style_cards)
        graph.add_node("rank_and_format", rank_and_format)

        graph.set_entry_point("prepare_context")
        graph.add_edge("prepare_context", "synthesize_trends")
        graph.add_edge("synthesize_trends", "generate_style_cards")
        graph.add_edge("generate_style_cards", "rank_and_format")
        graph.add_edge("rank_and_format", END)

        return graph.compile()

    except ImportError:
        # Fallback: simple sequential runner that mimics the graph
        class SequentialRunner:
            def invoke(self, state: AgentState) -> AgentState:
                state = prepare_context(state)
                state = synthesize_trends(state)
                state = generate_style_cards(state)
                state = rank_and_format(state)
                return state
        return SequentialRunner()


# ── High-level convenience function ──────────────────────────────────────────

def run_forecast(trend_reports: list, analysis_window_days: int = 30) -> ForecastReport:
    """
    One-call interface: takes trend reports → returns ForecastReport.

    Args:
        trend_reports: list[TrendReport] from TrendEngine
        analysis_window_days: lookback window used (for metadata)
    """
    from datetime import datetime

    agent = build_forecast_agent()

    initial_state: AgentState = {
        "trend_reports": trend_reports,
        "contexts": [],
        "raw_synthesis": "",
        "style_cards": [],
        "executive_summary": "",
        "macro_direction": "",
        "error": None,
    }

    result = agent.invoke(initial_state)

    rising = sum(1 for r in trend_reports if getattr(r, "velocity", "") == "rising")
    declining = sum(1 for r in trend_reports if getattr(r, "velocity", "") == "declining")

    return ForecastReport(
        generated_at=datetime.utcnow().isoformat() + "Z",
        analysis_window_days=analysis_window_days,
        total_trends_analyzed=len(trend_reports),
        rising_count=rising,
        declining_count=declining,
        executive_summary=result.get("executive_summary", ""),
        macro_direction=result.get("macro_direction", ""),
        style_cards=result.get("style_cards", []),
        raw_trend_contexts=result.get("contexts", []),
    )
