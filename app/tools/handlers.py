from __future__ import annotations

import json
import math
import os
import secrets
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Tuple

import openai

# Reuse existing OpenAI integration style
_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_DEFAULT_MODEL = "gpt-4.1-mini"
_DAILY_PHRASE_CACHE: Dict[str, Dict[str, str]] = {}


def run_resume_bullets(payload: Dict[str, Any]) -> Dict[str, str]:
    """Generate improved resume bullets."""
    experience_text = (payload.get("experience_text") or "").strip()
    target_role = (payload.get("target_role") or "").strip()
    job_description = (payload.get("job_description") or "").strip()
    tone = (payload.get("tone") or "Professional").strip() or "Professional"

    prompt = (
        "You are a concise resume assistant. Improve the provided experience notes into numbered bullets.\n"
        "- Return 6-10 numbered bullets only (e.g., '1. ...').\n"
        "- Use strong action verbs, keep one line each.\n"
        "- Add realistic metric placeholders if missing (e.g., 'reduced by X%').\n"
        "- Align with the target role and any job description keywords without inventing credentials.\n"
        "- Prefer IT support, security, automation, and troubleshooting emphasis if the target role suggests it.\n"
        f"Tone: {tone}\n"
        f"Target role: {target_role or 'General'}\n"
        f"Job description (optional): {job_description or 'N/A'}\n"
        f"Experience notes: {experience_text}\n"
        "Output only the bullets."
    )

    try:
        result = _client.chat.completions.create(
            model=_DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You improve resume bullets with brevity and measurable impact."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=650,
            temperature=0.2,
            timeout=15,
        )
        content = result.choices[0].message.content.strip()
        return {"output": content}
    except Exception as exc:
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc


def _parse_participants(raw: str) -> Tuple[List[str], Dict[str, str]]:
    names = [p.strip() for p in raw.split(",") if p.strip()]
    if not names:
        raise ValueError("Participants list is empty.")
    normalized = {n.lower(): n for n in names}
    return names, normalized


def _parse_expenses(raw: str, participants_map: Dict[str, str], all_participants: List[str]) -> List[Dict[str, Any]]:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        raise ValueError("No expenses provided.")
    if len(lines) > 200:
        raise ValueError("Too many expense lines (limit 200).")

    expenses: List[Dict[str, Any]] = []
    for idx, line in enumerate(lines, start=1):
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            raise ValueError(f"Line {idx}: expected at least payer | amount | description.")
        payer_raw, amount_raw, description = parts[:3]
        split_raw = parts[3] if len(parts) >= 4 else ""

        payer_norm = payer_raw.lower()
        if payer_norm not in participants_map:
            raise ValueError(f"Line {idx}: payer '{payer_raw}' not in participants.")

        try:
            amount = float(amount_raw)
        except Exception:
            raise ValueError(f"Line {idx}: amount '{amount_raw}' is not a number.")
        if amount <= 0:
            raise ValueError(f"Line {idx}: amount must be positive.")

        if split_raw:
            split_names = [p.strip() for p in split_raw.split(",") if p.strip()]
            if not split_names:
                raise ValueError(f"Line {idx}: split list is empty.")
            split_norm = []
            for name in split_names:
                key = name.lower()
                if key not in participants_map:
                    raise ValueError(f"Line {idx}: split participant '{name}' not in participants.")
                split_norm.append(participants_map[key])
        else:
            split_norm = list(all_participants)

        expenses.append(
            {
                "payer": participants_map[payer_norm],
                "amount": amount,
                "description": description,
                "split_with": split_norm,
            }
        )
    return expenses


def _compute_balances(participants: List[str], expenses: List[Dict[str, Any]]):
    balances = {p: {"paid": 0.0, "owes": 0.0} for p in participants}
    for exp in expenses:
        payer = exp["payer"]
        amount = exp["amount"]
        split = exp["split_with"]
        split_count = len(split)
        if split_count == 0:
            continue
        share = amount / split_count
        balances[payer]["paid"] += amount
        for person in split:
            balances[person]["owes"] += share
    return balances


def _compute_settlements(balances: Dict[str, Dict[str, float]]) -> List[Tuple[str, str, float]]:
    debtors = []
    creditors = []
    for name, data in balances.items():
        net = round(data["paid"] - data["owes"], 2)
        if math.isclose(net, 0, abs_tol=0.01):
            continue
        if net < 0:
            debtors.append([name, -net])  # amount owed
        else:
            creditors.append([name, net])  # amount to receive

    debtors.sort(key=lambda x: x[1])  # smallest debt first
    creditors.sort(key=lambda x: x[1], reverse=True)  # largest credit first

    settlements: List[Tuple[str, str, float]] = []
    i = 0
    j = 0
    while i < len(debtors) and j < len(creditors):
        debtor, debt_amt = debtors[i]
        creditor, cred_amt = creditors[j]
        pay = round(min(debt_amt, cred_amt), 2)
        if pay > 0:
            settlements.append((debtor, creditor, pay))
        debtors[i][1] = round(debt_amt - pay, 2)
        creditors[j][1] = round(cred_amt - pay, 2)
        if debtors[i][1] <= 0.01:
            i += 1
        if creditors[j][1] <= 0.01:
            j += 1
    return settlements


def run_expense_splitter(payload: Dict[str, Any]) -> Dict[str, Any]:
    event_name = (payload.get("event_name") or "").strip()
    participants_raw = payload.get("participants") or ""
    expenses_raw = payload.get("expenses") or ""

    participants, participants_map = _parse_participants(participants_raw)
    expenses = _parse_expenses(expenses_raw, participants_map, participants)
    balances = _compute_balances(participants, expenses)
    settlements = _compute_settlements(balances)

    # Build output text
    lines: List[str] = []
    lines.append(f"Event: {event_name}")
    lines.append(f"Participants: {', '.join(participants)}")
    lines.append("")
    lines.append("Totals:")
    for name in participants:
        data = balances[name]
        paid = round(data["paid"], 2)
        owes = round(data["owes"], 2)
        net = round(paid - owes, 2)
        lines.append(f"{name} paid {paid:.2f}, owes {owes:.2f}, net {net:+.2f}")
    lines.append("")
    lines.append("Settle:")
    if settlements:
        for debtor, creditor, amt in settlements:
            lines.append(f"{debtor} pays {creditor} {amt:.2f}")
    else:
        lines.append("No transfers needed.")

    output_text = "\n".join(lines)
    token = secrets.token_urlsafe(8)
    expires_at = datetime.now(timezone.utc) + timedelta(days=14)

    payload_json = {
        "type": "expense-splitter",
        "event_name": event_name,
        "participants": participants,
        "expenses_raw": expenses_raw,
        "settlement_text": output_text,
    }

    return {
        "output": output_text,
        "token": token,
        "expires_at": expires_at.isoformat(),
        "payload_json": json.dumps(payload_json),
    }


# ---- Trip Planner ----


def _coerce_amount(val, label):
    try:
        amt = float(val)
    except Exception:
        raise ValueError(f"{label} must be a number.")
    if amt < 0:
        raise ValueError(f"{label} must be positive.")
    return amt


def run_trip_planner(payload: Dict[str, Any]) -> Dict[str, Any]:
    trip_name = (payload.get("trip_name") or "").strip()
    currency = (payload.get("currency") or "").strip() or "USD"
    notes = (payload.get("notes") or "").strip()
    people = payload.get("people") or []
    budgets = payload.get("budgets") or []
    expenses_paid = payload.get("expenses_paid") or []
    items_planned = payload.get("items_planned") or []

    if not trip_name:
        raise ValueError('"trip_name" is required.')
    if not currency:
        raise ValueError('"currency" is required.')
    if not isinstance(people, list) or len(people) == 0:
        raise ValueError("At least one person is required.")
    people_norm = [p.strip() for p in people if p and str(p).strip()]
    if not people_norm:
        raise ValueError("At least one person is required.")
    people_lower_map = {p.lower(): p for p in people_norm}

    # Budgets
    budget_summaries = []
    total_budget = 0.0
    for b in budgets:
        cat = (b.get("category") or "Other").strip() or "Other"
        amt_raw = b.get("amount", 0)
        amt = _coerce_amount(amt_raw, f"Budget for {cat}")
        total_budget += amt
        budget_summaries.append({"category": cat, "amount": amt})

    # Expenses (paid)
    balances = {p: {"paid": 0.0, "owes": 0.0} for p in people_norm}
    expenses_clean = []
    for idx, exp in enumerate(expenses_paid, start=1):
        payer_raw = (exp.get("payer") or "").strip()
        if not payer_raw:
            raise ValueError(f"Expense {idx}: payer required.")
        payer_key = payer_raw.lower()
        if payer_key not in people_lower_map:
            raise ValueError(f"Expense {idx}: payer '{payer_raw}' not in people list.")
        payer = people_lower_map[payer_key]
        amount = _coerce_amount(exp.get("amount", 0), f"Expense {idx} amount")
        if amount <= 0:
            raise ValueError(f"Expense {idx}: amount must be positive.")
        category = (exp.get("category") or "").strip() or "Other"
        description = (exp.get("description") or "").strip()
        split_with = exp.get("split_with") or []
        if not split_with:
            split_with = list(people_norm)
        split_clean = []
        for name in split_with:
            key = (name or "").strip().lower()
            if key not in people_lower_map:
                raise ValueError(f"Expense {idx}: split participant '{name}' not in people list.")
            split_clean.append(people_lower_map[key])
        share = amount / len(split_clean)
        balances[payer]["paid"] += amount
        for person in split_clean:
            balances[person]["owes"] += share
        expenses_clean.append(
            {
                "payer": payer,
                "amount": amount,
                "category": category,
                "description": description,
                "split_with": split_clean,
                "status": "Paid",
            }
        )

    # Planned items
    planned_clean = []
    for idx, item in enumerate(items_planned, start=1):
        category = (item.get("category") or "").strip() or "Other"
        amount = _coerce_amount(item.get("amount", 0), f"Planned item {idx} amount")
        description = (item.get("description") or "").strip()
        due_date = (item.get("due_date") or "").strip()
        assigned_to_raw = (item.get("assigned_to") or "").strip()
        assigned_to = None
        if assigned_to_raw:
            key = assigned_to_raw.lower()
            if key not in people_lower_map:
                raise ValueError(f"Planned item {idx}: assigned_to '{assigned_to_raw}' not in people list.")
            assigned_to = people_lower_map[key]
        planned_clean.append(
            {
                "category": category,
                "amount": amount,
                "description": description,
                "due_date": due_date,
                "assigned_to": assigned_to,
                "status": "Planned",
            }
        )

    settlements = _compute_settlements(balances)

    per_person = []
    for person, data in balances.items():
        paid = round(data["paid"], 2)
        owes = round(data["owes"], 2)
        net = round(paid - owes, 2)
        per_person.append({"name": person, "paid": paid, "owes": owes, "net": net})

    total_paid = sum(d["paid"] for d in balances.values())
    total_owes = sum(d["owes"] for d in balances.values())

    # Budget remaining
    spent_by_category = {}
    for exp in expenses_clean:
        cat = exp["category"]
        spent_by_category[cat] = spent_by_category.get(cat, 0) + exp["amount"]
    planned_by_category = {}
    for item in planned_clean:
        cat = item["category"]
        planned_by_category[cat] = planned_by_category.get(cat, 0) + item["amount"]

    budget_summary = []
    all_categories = set(spent_by_category.keys()) | set(planned_by_category.keys()) | {b["category"] for b in budget_summaries}
    budget_map = {b["category"]: b["amount"] for b in budget_summaries}
    for cat in all_categories:
        planned_amt = budget_map.get(cat, 0.0)
        spent = spent_by_category.get(cat, 0.0)
        upcoming = planned_by_category.get(cat, 0.0)
        remaining = planned_amt - (spent + upcoming)
        budget_summary.append(
            {
                "category": cat,
                "planned": round(planned_amt, 2),
                "spent": round(spent, 2),
                "upcoming": round(upcoming, 2),
                "remaining": round(remaining, 2),
            }
        )

    lines: List[str] = []
    lines.append(f"Trip: {trip_name} ({currency})")
    lines.append(f"People: {', '.join(people_norm)}")
    if notes:
        lines.append(f"Notes: {notes}")
    lines.append("")
    lines.append("Budgets:")
    if budget_summary:
        for b in budget_summary:
            lines.append(f"{b['category']}: planned {b['planned']:.2f}, spent {b['spent']:.2f}, upcoming {b['upcoming']:.2f}, remaining {b['remaining']:.2f}")
    else:
        lines.append("None set.")
    lines.append("")
    lines.append("Totals:")
    for p in per_person:
        lines.append(f"{p['name']}: paid {p['paid']:.2f}, owes {p['owes']:.2f}, net {p['net']:+.2f}")
    lines.append("")
    lines.append("Settle (paid items only):")
    if settlements:
        for debtor, creditor, amt in settlements:
            lines.append(f"{debtor} pays {creditor} {amt:.2f}")
    else:
        lines.append("No transfers needed.")

    output_text = "\n".join(lines)
    token = secrets.token_urlsafe(8)
    expires_at = datetime.now(timezone.utc) + timedelta(days=14)

    payload_json = {
        "type": "trip-planner",
        "trip_name": trip_name,
        "currency": currency,
        "notes": notes,
        "people": people_norm,
        "budgets": budget_summaries,
        "expenses_paid": expenses_clean,
        "items_planned": planned_clean,
        "settlement_text": output_text,
        "per_person": per_person,
        "settlement_transfers": settlements,
        "budget_summary": budget_summary,
        "total_budget": round(total_budget, 2),
        "total_paid": round(total_paid, 2),
        "total_owes": round(total_owes, 2),
    }

    return {
        "output": output_text,
        "token": token,
        "expires_at": expires_at.isoformat(),
        "payload_json": json.dumps(payload_json),
        "structured": {
            "per_person": per_person,
            "settlement_transfers": settlements,
            "budget_summary": budget_summary,
            "total_budget": round(total_budget, 2),
            "total_paid": round(total_paid, 2),
            "total_owes": round(total_owes, 2),
        },
    }


# ---- Daily Phrase ----


def run_daily_phrase(payload: Dict[str, Any]) -> Dict[str, Any]:
    language = (payload.get("language") or "Spanish").strip()
    level = (payload.get("level") or "Beginner").strip()
    today = datetime.now().strftime("%Y-%m-%d")
    cache_key = f"daily_phrase::{language}::{level}::{today}"

    if cache_key in _DAILY_PHRASE_CACHE:
        cached = _DAILY_PHRASE_CACHE[cache_key]
        return {
            "phrase": cached["phrase"],
            "translation": cached["translation"],
            "example": cached["example"],
            "date": today,
        }

    prompt = (
        "Provide one useful daily phrase.\n"
        f"Language: {language}\n"
        f"Level: {level}\n"
        "Return three parts:\n"
        "- Phrase (short; for Beginner keep it simple, avoid long sentences)\n"
        "- Translation (clear English)\n"
        "- Example (natural sentence using the phrase)\n"
        "No slang unless level is Advanced. No emojis. No markdown. Keep it concise."
    )

    try:
        result = _client.chat.completions.create(
            model=_DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": "You are a concise language tutor."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.4,
            timeout=15,
        )
        content = result.choices[0].message.content.strip()
        # naive parse: expect three lines
        parts = [p.strip() for p in content.split("\n") if p.strip()]
        phrase = parts[0] if parts else content
        translation = parts[1] if len(parts) > 1 else ""
        example = parts[2] if len(parts) > 2 else ""
        data = {"phrase": phrase, "translation": translation, "example": example, "date": today}
        _DAILY_PHRASE_CACHE[cache_key] = data
        return data
    except Exception as exc:
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc
