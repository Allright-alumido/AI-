"""
explain.py

Turns a recommendation dict (from inference.Recommender.recommend_from_history
/ recommend_manual) into:
  - a numeric "decision breakdown" (compute_decision_breakdown) -- the
    Demand P90 / Current Stock / Safety Buffer / Expected Ending Inventory
    figures shown in the UI, always available with no LLM/network involved.
  - a confidence rating (confidence_level) based on how wide the gap is
    between the mean and P90 demand forecast.
  - a plain-language explanation (explain_recommendation) for a store
    manager, via an LLM if one is configured, or a template otherwise.

Note on the decision breakdown: the RL policy is a learned neural network,
not literally this formula -- "Safety Buffer" and "Expected Ending
Inventory" here are a business-friendly *approximation* of its reasoning,
computed directly from the same forecast/inventory numbers the policy saw,
so a manager can sanity-check the recommendation without needing to
understand PPO. It's intentionally the same arithmetic a simple newsvendor
rule-of-thumb would use, not a trace of the model's internal computation.

explain_recommendation() checks for an API key for Anthropic
(ANTHROPIC_API_KEY), Gemini (GOOGLE_API_KEY / GEMINI_API_KEY), or OpenAI
(OPENAI_API_KEY), in that order, tries whichever is configured, and
silently falls back to a deterministic template built straight from the
numbers if no key is set, the matching package isn't installed, or the
call fails for any reason. It never raises -- safe to call from a UI
without extra error handling. It returns (text, source) where source is
one of "anthropic", "gemini", "openai", "template" so the UI can show
which one actually produced the text.

Where each key can live (checked in this order via _get_api_key):
  1. OS environment variable (os.environ) -- convenient for local dev/testing
     from a terminal, e.g. `$env:ANTHROPIC_API_KEY = "..."` before
     `streamlit run app.py`.
  2. Streamlit secrets (.streamlit/secrets.toml), i.e. st.secrets -- the
     friendlier option for an actually-deployed app: the developer sets the
     key ONCE in that file at deploy time, and end users never see, enter,
     or need to know an API key exists at all. Reading st.secrets is wrapped
     in a try/except and only attempted if `streamlit` is importable, so
     this module stays fully usable (and unit-testable) in a plain Python
     environment with no streamlit installed and no secrets.toml present --
     it just silently skips to the template fallback in that case.
"""

from __future__ import annotations

import os

from reward_configs import SCENARIOS


def _get_api_key(*names: str) -> str | None:
    """Look up any of `names` as an API key, OS environment first, then
    Streamlit's st.secrets if available (see module docstring). Returns the
    first non-empty match, or None if nothing is configured anywhere."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    try:
        import streamlit as st
        for name in names:
            value = st.secrets.get(name)
            if value:
                return value
    except Exception:
        # streamlit not installed, no secrets.toml configured, or running
        # outside a streamlit app context -- fall through silently.
        pass
    return None


# --------------------------------------------------------------------- #
# Numeric decision breakdown (no LLM, always available)
# --------------------------------------------------------------------- #
def compute_decision_breakdown(rec: dict) -> dict:
    """
    Safety Buffer   = order_qty - (Demand P90 - Current Stock)
                      i.e. how much cushion the order leaves beyond just
                      covering the worst-case (P90) demand gap. Negative
                      means the order doesn't fully cover the P90 case.
    Expected Ending Inventory = Current Stock + order_qty - Demand Mean
                      i.e. where inventory is expected to land assuming
                      average (not worst-case) demand shows up.
    """
    current_stock = rec["current_inventory"]
    demand_p90 = rec["forecast_q90"]
    demand_mean = rec["forecast_mean"]
    order_qty = rec["fulfilled_order_qty"]
    safety_buffer = order_qty - (demand_p90 - current_stock)
    expected_ending_inventory = current_stock + order_qty - demand_mean
    return {
        "order_qty": order_qty,
        "demand_mean": demand_mean,
        "demand_p90": demand_p90,
        "current_stock": current_stock,
        "safety_buffer": safety_buffer,
        "expected_ending_inventory": expected_ending_inventory,
    }


def breakdown_text(rec: dict) -> str:
    b = compute_decision_breakdown(rec)
    covers_p90 = b["safety_buffer"] >= 0
    note = "" if covers_p90 else " (does not fully cover the worst-case P90 demand)"
    return (
        f"Recommended {b['order_qty']:.0f} because: "
        f"Demand P90 = {b['demand_p90']:.0f}, Current Stock = {b['current_stock']:.0f}, "
        f"Safety Buffer = {b['safety_buffer']:.0f}{note}, "
        f"Expected Ending Inventory = {b['expected_ending_inventory']:.0f}."
    )


# --------------------------------------------------------------------- #
# Confidence rating
# --------------------------------------------------------------------- #
def confidence_level(rec: dict) -> dict:
    """Based on the relative gap between mean and P90 demand forecast.
    Thresholds (15% / 35% of the mean) are reasonable defaults -- tune
    them here if they don't match how volatile your actual demand is."""
    mean = max(rec["forecast_mean"], 1e-6)
    gap = max(rec["forecast_q90"] - rec["forecast_mean"], 0.0)
    ratio = gap / mean
    if ratio < 0.15:
        label, color = "High Confidence", "green"
    elif ratio < 0.35:
        label, color = "Medium Confidence", "orange"
    else:
        label, color = "Low Confidence", "red"
    score = max(0.0, min(1.0, 1.0 - ratio))
    return {"label": label, "color": color, "score": score, "gap_ratio": ratio}


# --------------------------------------------------------------------- #
# Plain-language explanation
# --------------------------------------------------------------------- #
def _template_explanation(rec: dict) -> str:
    scenario_desc = SCENARIOS[rec["scenario"]].description
    lines = [
        breakdown_text(rec),
        f"(Policy asked for {rec['requested_order_qty']:.0f}"
        + (", capped by warehouse capacity)" if rec.get("capped_by_capacity") else ")"),
        f"Strategy in use -- {rec['scenario_label']}: {scenario_desc}",
    ]
    if "historical_order_qty" in rec:
        lines.append(
            f"For comparison, the historical order on this date was "
            f"{rec['historical_order_qty']:.0f} units."
        )
    return "\n".join(lines)


def _prompt_for(rec: dict) -> str:
    scenario_desc = SCENARIOS[rec["scenario"]].description
    b = compute_decision_breakdown(rec)
    extra = (
        f"For comparison, the historical order on this date was {rec['historical_order_qty']:.0f} units.\n"
        if "historical_order_qty" in rec else ""
    )
    return (
        "You are helping a retail store manager understand an automated restocking "
        "recommendation. Write a short (2-4 sentence), plain-language explanation of why "
        "this recommendation makes sense, in a friendly, confident tone -- no jargon like "
        "'RL', 'policy', or 'reward function'. You MUST reference these exact figures so "
        "the explanation matches the numbers already shown on screen: Demand P90 = "
        f"{b['demand_p90']:.0f}, Current Stock = {b['current_stock']:.0f}, Safety Buffer = "
        f"{b['safety_buffer']:.0f}, Expected Ending Inventory = {b['expected_ending_inventory']:.0f}.\n\n"
        f"Store: {rec.get('store_id')}, Product: {rec.get('product_id')}, Date: {rec.get('date')}\n"
        f"Recommended order: {rec['fulfilled_order_qty']:.0f} units\n"
        f"Forecasted demand: ~{b['demand_mean']:.0f} units (up to {b['demand_p90']:.0f} in a strong-demand scenario)\n"
        f"Current inventory on hand: {b['current_stock']:.0f} units\n"
        f"Strategy selected: \"{rec['scenario_label']}\" -- {scenario_desc}\n"
        f"{extra}"
    )


def _try_anthropic(prompt: str) -> str | None:
    api_key = _get_api_key("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception:
        return None


def _try_gemini(prompt: str) -> str | None:
    """Uses Google's current unified SDK (package `google-genai`, imported as
    `google.genai`). The older `google-generativeai` package (imported as
    `google.generativeai`, with `genai.configure()` / `GenerativeModel(...)`)
    is fully deprecated as of 2026 and no longer maintained -- if you installed
    that one instead, `pip uninstall google-generativeai` and
    `pip install google-genai`.
    """
    api_key = _get_api_key("GOOGLE_API_KEY", "GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        # Try the current flagship Flash model first, then a second,
        # typically-less-congested free-tier model if that one is briefly
        # unavailable (Google's free tier returns 503 UNAVAILABLE when a
        # specific model is under heavy demand -- transient, not a bug, and
        # a different model ID often has spare capacity at the same moment).
        # Update these if either has itself been retired/renamed by the time
        # you run this -- check ai.google.dev/gemini-api/docs/models for the
        # current stable, free-tier-eligible model names. (Google also
        # restricts some older model IDs, e.g. gemini-2.5-flash, to existing
        # projects only -- new API keys get a 404 "no longer available to new
        # users" even though the model is still listed in the docs.)
        for model_name in ("gemini-3.5-flash", "gemini-3.1-flash-lite"):
            try:
                resp = client.models.generate_content(model=model_name, contents=prompt)
                return resp.text.strip()
            except Exception:
                continue
        return None
    except Exception:
        return None


def _try_openai(prompt: str) -> str | None:
    api_key = _get_api_key("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


def explain_recommendation(rec: dict, use_llm: bool = True) -> tuple[str, str]:
    """Returns (explanation_text, source) where source is 'anthropic',
    'gemini', 'openai', or 'template'."""
    if use_llm:
        prompt = _prompt_for(rec)
        for name, attempt in (("anthropic", _try_anthropic), ("gemini", _try_gemini), ("openai", _try_openai)):
            result = attempt(prompt)
            if result:
                return result, name
    return _template_explanation(rec), "template"
