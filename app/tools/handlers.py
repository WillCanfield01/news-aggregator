from __future__ import annotations

import json
import math
import os
import re
import secrets
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Tuple

import openai

# Reuse existing OpenAI integration style
_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_DEFAULT_MODEL = "gpt-4.1-mini"
_DAILY_PHRASE_CACHE: Dict[str, Dict[str, str]] = {}

def _parse_decision_options(raw: str) -> List[Dict[str, str]]:
    lines = [line.strip() for line in (raw or "").splitlines() if line.strip()]
    options: List[Dict[str, str]] = []
    for line in lines:
        # Format A: Name | pros | cons
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2:
            name = parts[0]
            pros = parts[1] if len(parts) >= 2 else ""
            cons = parts[2] if len(parts) >= 3 else ""
            options.append({"name": name, "pros": pros, "cons": cons})
            continue

        # Format B: Name, pros: ..., cons: ...
        match = re.match(r"^(?P<name>[^,]+),\s*pros:\s*(?P<pros>.*?)(?:,\s*cons:\s*(?P<cons>.*))?$", line, re.I)
        if match:
            options.append(
                {
                    "name": match.group("name").strip(),
                    "pros": (match.group("pros") or "").strip(),
                    "cons": (match.group("cons") or "").strip(),
                }
            )
            continue

        # Fallback: treat as name only
        options.append({"name": line, "pros": "", "cons": ""})

    return options


def _clean_daily_phrase_value(value: Any) -> str:
    text = str(value or "").strip().strip("\"'")
    if not text:
        return ""
    lowered = text.lower()
    for label in ("phrase:", "translation:", "example:"):
        if lowered.startswith(label):
            text = text.split(":", 1)[1].strip()
            break
    text = re.sub(r"^[\-\u2013\u2014]\s*", "", text)
    text = re.sub(r"^\d+\.\s*", "", text)
    return text.strip()


def _parse_daily_phrase_content(content: str) -> Dict[str, str]:
    raw = (content or "").strip()
    if not raw:
        return {"phrase": "", "translation": "", "example": ""}

    parsed: Dict[str, Any] | None = None
    try:
        parsed = json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.S)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except Exception:
                parsed = None

    if isinstance(parsed, dict):
        return {
            "phrase": _clean_daily_phrase_value(parsed.get("phrase")),
            "translation": _clean_daily_phrase_value(parsed.get("translation")),
            "example": _clean_daily_phrase_value(parsed.get("example")),
        }

    parts = [p.strip() for p in raw.splitlines() if p.strip()]
    phrase = parts[0] if parts else raw
    translation = parts[1] if len(parts) > 1 else ""
    example = parts[2] if len(parts) > 2 else ""
    return {
        "phrase": _clean_daily_phrase_value(phrase),
        "translation": _clean_daily_phrase_value(translation),
        "example": _clean_daily_phrase_value(example),
    }


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
        "Respond ONLY with strict JSON in this exact shape (no markdown, no labels, no extra text):\n"
        '{"phrase":"<phrase in the target language>","translation":"<english translation>","example":"<natural sentence using the phrase>"}\n'
        "Keep the phrase short for beginners, avoid slang unless Advanced, and keep the example concise."
    )

    try:
        result = _client.chat.completions.create(
            model=_DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise language tutor. Reply with JSON only using keys phrase, translation, example.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.4,
            timeout=15,
        )
        content = (result.choices[0].message.content or "").strip()
        parsed = _parse_daily_phrase_content(content)
        data = {
            "phrase": parsed.get("phrase", ""),
            "translation": parsed.get("translation", ""),
            "example": parsed.get("example", ""),
            "date": today,
        }
        _DAILY_PHRASE_CACHE[cache_key] = data
        return data
    except Exception as exc:
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc


# ---- Decision Helper ----


def run_decision_helper(payload: Dict[str, Any]) -> Dict[str, Any]:
    decision = (payload.get("decision") or "").strip()
    options_raw = payload.get("options") or ""
    priority = (payload.get("priority") or "Balanced").strip() or "Balanced"
    constraints = (payload.get("constraints") or "").strip()

    if not decision:
        raise ValueError('"decision" is required.')

    parsed_options = _parse_decision_options(options_raw)
    parsed_options = [opt for opt in parsed_options if opt.get("name")]
    if len(parsed_options) < 2:
        raise ValueError("Please provide at least two options (one per line).")

    options_text = "\n".join(
        f"- {opt['name']}\n  pros: {opt.get('pros', '') or '—'}\n  cons: {opt.get('cons', '') or '—'}"
        for opt in parsed_options
    )

    prompt = (
        "You are a concise decision coach.\n"
        "Restate the decision in one line.\n"
        "List the options with pros/cons in a compact table-like summary.\n"
        "If and only if critical info is missing, ask up to 2 short clarifying questions; otherwise proceed.\n"
        "Provide a clear recommendation aligned to the stated priority and constraints.\n"
        "Add 3 quick next steps.\n"
        "Keep it brief and scannable.\n"
        f"Decision: {decision}\n"
        f"Priority: {priority}\n"
        f"Constraints: {constraints or 'None provided'}\n"
        "Options:\n"
        f"{options_text}"
    )

    try:
        result = _client.chat.completions.create(
            model=_DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a structured decision helper. Be concise, neutral, and practical. Never output markdown tables.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=650,
            temperature=0.25,
            timeout=15,
        )
        content = (result.choices[0].message.content or "").strip()
        output_lines = [
            f"Decision: {decision}",
            "",
            "Options:",
            content,
        ]
        return {"output": "\n".join(output_lines)}
    except Exception as exc:
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc


# ---- Worth It Calculator ----


def _parse_money(val: str) -> float:
    try:
        clean = re.sub(r"[,$]", "", str(val)).strip()
        if clean.startswith("$"):
            clean = clean[1:]
        amount = float(clean)
    except Exception:
        raise ValueError("Enter cost as a number (e.g., 25 or 25.99).")
    if amount <= 0:
        raise ValueError("Cost must be greater than zero.")
    return amount


def _parse_hours(val: str) -> float:
    try:
        hours = float(str(val).strip())
    except Exception:
        raise ValueError("Enter hours as a number (e.g., 2 or 2.5).")
    if hours <= 0:
        raise ValueError("Hours must be greater than zero.")
    return hours


def _parse_timeframe(raw: str, frequency: str) -> tuple[float, str]:
    raw = (raw or "").strip().lower()
    if not raw:
        if frequency == "Weekly":
            return 4.0, "weeks"
        if frequency == "Monthly":
            return 1.0, "months"
        return 0.0, ""

    match = re.match(r"^(\d+(?:\.\d+)?)(\s*(weeks?|w|months?|m))$", raw)
    if not match:
        raise ValueError("Timeframe must be like '4 weeks' or '3 months'.")
    amount = float(match.group(1))
    unit_raw = match.group(2).strip()
    if amount <= 0:
        raise ValueError("Timeframe must be greater than zero.")
    if unit_raw.startswith("w"):
        return amount, "weeks"
    if unit_raw.startswith("m"):
        return amount, "months"
    raise ValueError("Timeframe must specify weeks or months.")


def _format_currency(val: float) -> str:
    return f"${val:,.2f}"


def _format_hours(val: float) -> str:
    return f"{val:,.2f}".rstrip("0").rstrip(".")


def run_worth_it(payload: Dict[str, Any]) -> Dict[str, Any]:
    item_name = (payload.get("item_name") or "").strip()
    mode = (payload.get("mode") or "enjoyment").strip().lower()
    cost_raw = payload.get("cost") or payload.get("cost_amount") or ""
    frequency = (payload.get("frequency") or "One-time").strip()
    minutes_enjoyment_raw = payload.get("minutes_per_occurrence") or ""
    minutes_saved_raw = payload.get("minutes_saved_per_use") or ""
    hourly_value_raw = payload.get("hourly_value") or ""
    target_value_raw = payload.get("target_value_per_hour") or ""
    expected_uses_raw = payload.get("expected_uses_per_month") or ""
    timeframe_months_raw = payload.get("timeframe_months") or payload.get("timeframe") or ""
    notes = (payload.get("notes") or "").strip()
    compare_enabled = (payload.get("compare_enabled") or "").lower() in {"yes", "true", "1", "on"}
    option_b_name = (payload.get("option_b_name") or "").strip()
    option_b_cost = payload.get("option_b_cost") or ""
    option_b_minutes = payload.get("option_b_minutes") or ""
    option_b_frequency = (payload.get("option_b_frequency") or "").strip()
    option_b_hourly_raw = payload.get("option_b_hourly_value") or ""
    nudge_enabled = (payload.get("nudge_enabled") or "").lower() in {"yes", "true", "1", "on"}
    return_window_raw = payload.get("return_window_days") or ""
    quit_chance_raw = payload.get("quit_chance_percent") or ""
    follow_through_raw = payload.get("follow_through_percent") or ""
    nudge_enabled = (payload.get("nudge_enabled") or "").lower() in {"yes", "true", "1", "on"}
    return_window_raw = payload.get("return_window_days") or ""
    quit_chance_raw = payload.get("quit_chance_percent") or ""
    follow_through_raw = payload.get("follow_through_percent") or ""

    def _parse_positive_float(value: Any, label: str, allow_zero: bool = False) -> float:
        try:
            num = float(str(value).replace(",", "").strip())
        except Exception:
            raise ValueError(f"{label} must be a number.")
        if not allow_zero and num <= 0:
            raise ValueError(f"{label} must be greater than zero.")
        if allow_zero and num < 0:
            raise ValueError(f"{label} cannot be negative.")
        return num

    cost = _parse_positive_float(cost_raw, "Cost")

    frequency_options = {"One-time", "Daily", "Weekly", "Biweekly", "Monthly", "Yearly"}
    if frequency not in frequency_options:
        raise ValueError("Frequency must be One-time, Daily, Weekly, Biweekly, Monthly, or Yearly.")

    sessions_per_month_map = {
        "Daily": 30.0,
        "Weekly": 4.345,
        "Biweekly": 2.1725,
        "Monthly": 1.0,
        "Yearly": 1.0 / 12.0,
        "One-time": 0.0,
    }
    sessions_per_month = sessions_per_month_map.get(frequency, 0.0)

    if str(timeframe_months_raw).strip():
        timeframe_months = _parse_positive_float(timeframe_months_raw, "Timeframe (months)", allow_zero=False)
    else:
        timeframe_defaults = {
            "Yearly": 12.0,
            "Monthly": 1.0,
            "Weekly": 1.0,
            "Daily": 1.0,
            "Biweekly": 1.0,
            "One-time": 1.0,
        }
        timeframe_months = timeframe_defaults.get(frequency, 1.0)

    if mode not in {"enjoyment", "time_saved"}:
        raise ValueError("Mode must be enjoyment or time_saved.")

    share_card: Dict[str, Any]
    details: List[Dict[str, Any]] = []

    comparison = None
    metric_a_primary = None
    metric_a_break_even = None

    nudge_data = None

    if mode == "enjoyment":
        if not minutes_enjoyment_raw:
            raise ValueError("Minutes per occurrence is required.")
        minutes_per_occurrence = _parse_positive_float(minutes_enjoyment_raw, "Minutes per occurrence")
        target_value = (
            _parse_positive_float(target_value_raw, "Target value per hour")
            if str(target_value_raw).strip()
            else 25.0
        )
        sessions_total = (sessions_per_month * timeframe_months) or 1.0
        total_hours = sessions_total * (minutes_per_occurrence / 60.0)
        cost_per_hour = cost / max(total_hours, 0.01)
        metric_a_primary = cost_per_hour
        total_hours_needed = cost / max(target_value, 0.01)
        sessions_needed = total_hours_needed / max(minutes_per_occurrence / 60.0, 0.01)
        sessions_per_month_needed = sessions_needed / max(timeframe_months, 1.0)
        sessions_per_week_needed = sessions_per_month_needed / 4.345
        metric_a_break_even = sessions_per_month_needed

        if cost_per_hour <= target_value:
            verdict = "Worth it"
            pill_class = "success"
        elif cost_per_hour <= target_value * 1.5:
            verdict = "Maybe"
            pill_class = "warn"
        else:
            verdict = "Skip for now"
            pill_class = "danger"

        share_card = {
            "title": item_name or "Is this worth it?",
            "primary_metric_label": "Cost per hour",
            "primary_metric_value": _format_currency(cost_per_hour),
            "break_even_label": "Break-even usage",
            "break_even_value": f"{sessions_per_week_needed:,.2f}/week • {sessions_per_month_needed:,.2f}/month",
            "verdict": verdict,
            "pill_class": pill_class,
            "subtext": f"Target value: ${target_value:,.2f}/hr",
        }

        details = [
            {"label": "Frequency", "value": frequency},
            {"label": "Timeframe (months)", "value": _format_hours(timeframe_months)},
            {"label": "Sessions total", "value": _format_hours(sessions_total)},
            {"label": "Total hours", "value": _format_hours(total_hours)},
            {"label": "Cost", "value": _format_currency(cost)},
        ]
        if notes:
            details.append({"label": "Assumptions", "value": notes})

    else:
        # time_saved
        if not minutes_saved_raw:
            raise ValueError("Minutes saved per use is required.")
        if not str(hourly_value_raw).strip():
            raise ValueError("Hourly value is required for time_saved mode.")
        minutes_saved_per_use = _parse_positive_float(minutes_saved_raw, "Minutes saved per use")
        hourly_value = _parse_positive_float(hourly_value_raw, "Your hourly value")

        value_per_use = (minutes_saved_per_use / 60.0) * hourly_value
        break_even_uses = math.ceil(cost / max(value_per_use, 0.01))
        metric_a_primary = value_per_use
        metric_a_break_even = break_even_uses

        effective_cph_saved = None
        if str(expected_uses_raw).strip():
            expected_uses = _parse_positive_float(expected_uses_raw, "Expected uses per month")
            total_uses = expected_uses * timeframe_months
            total_saved_hours = total_uses * (minutes_saved_per_use / 60.0)
            effective_cph_saved = cost / max(total_saved_hours, 0.01)

        if value_per_use >= hourly_value * 0.25 or break_even_uses <= 10:
            verdict = "Worth it"
            pill_class = "success"
        elif break_even_uses <= 30 or value_per_use >= hourly_value * 0.1:
            verdict = "Maybe"
            pill_class = "warn"
        else:
            verdict = "Skip for now"
            pill_class = "danger"

        primary_label = "Value per use"
        primary_value = _format_currency(value_per_use)
        break_even_label = "Break-even uses"
        break_even_value = f"{break_even_uses}"

        if effective_cph_saved is not None:
            primary_label = "Cost per hour saved"
            primary_value = _format_currency(effective_cph_saved)

        share_card = {
            "title": item_name or "Is this worth it?",
            "primary_metric_label": primary_label,
            "primary_metric_value": primary_value,
            "break_even_label": break_even_label,
            "break_even_value": break_even_value,
            "verdict": verdict,
            "pill_class": pill_class,
            "subtext": f"Value per use: {_format_currency(value_per_use)}",
        }

        details = [
            {"label": "Frequency", "value": frequency},
            {"label": "Timeframe (months)", "value": _format_hours(timeframe_months)},
            {"label": "Break-even uses", "value": f"{break_even_uses}"},
            {"label": "Minutes saved per use", "value": _format_hours(minutes_saved_per_use)},
            {"label": "Hourly value", "value": _format_currency(hourly_value)},
        ]
        if effective_cph_saved is not None:
            details.append({"label": "Cost per hour saved", "value": _format_currency(effective_cph_saved)})
        if notes:
            details.append({"label": "Assumptions", "value": notes})

    if nudge_enabled:
        ft = _parse_positive_float(follow_through_raw or 100, "Follow-through percent", allow_zero=True) if str(follow_through_raw).strip() else 100.0
        quit_p = _parse_positive_float(quit_chance_raw or 0, "Quit chance percent", allow_zero=True) if str(quit_chance_raw).strip() else 0.0
        ft = min(max(ft, 0.0), 100.0)
        quit_p = min(max(quit_p, 0.0), 100.0)
        expected_usage_multiplier = (ft / 100.0) * (1 - quit_p / 100.0)
        expected_usage_multiplier = max(min(expected_usage_multiplier, 1.0), 0.05)

        if mode == "enjoyment":
            adjusted_metric = cost_per_hour / expected_usage_multiplier
            if adjusted_metric <= target_value:
                verdict = "Worth it"
                pill_class = "success"
            elif adjusted_metric <= target_value * 1.5:
                verdict = "Maybe"
                pill_class = "warn"
            else:
                verdict = "Skip for now"
                pill_class = "danger"
            share_card["primary_metric_value"] = _format_currency(adjusted_metric)
            share_card["verdict"] = verdict
            share_card["pill_class"] = pill_class
            share_card["subtext"] = f"Adjusted for follow-through • Target: ${target_value:,.2f}/hr"
            nudge_data = {
                "enabled": True,
                "expected_usage_multiplier": round(expected_usage_multiplier, 3),
                "adjusted_primary_metric_label": "Adjusted cost per hour",
                "adjusted_primary_metric_value": _format_currency(adjusted_metric),
                "note": f"Includes follow-through {ft:.0f}% and quit chance {quit_p:.0f}%",
                "return_window_days": return_window_raw.strip() if str(return_window_raw).strip() else "",
            }
        else:
            adjusted_break_even = math.ceil(break_even_uses / expected_usage_multiplier)
            if adjusted_break_even <= 10 or value_per_use >= hourly_value * 0.25:
                verdict = "Worth it"
                pill_class = "success"
            elif adjusted_break_even <= 30:
                verdict = "Maybe"
                pill_class = "warn"
            else:
                verdict = "Skip for now"
                pill_class = "danger"
            share_card["verdict"] = verdict
            share_card["pill_class"] = pill_class
            share_card["break_even_value"] = f"{adjusted_break_even}"
            share_card["subtext"] = "Adjusted for follow-through"
            nudge_data = {
                "enabled": True,
                "expected_usage_multiplier": round(expected_usage_multiplier, 3),
                "adjusted_primary_metric_label": "Adjusted break-even uses",
                "adjusted_primary_metric_value": f"{adjusted_break_even}",
                "note": f"Includes follow-through {ft:.0f}% and quit chance {quit_p:.0f}%",
                "return_window_days": return_window_raw.strip() if str(return_window_raw).strip() else "",
            }

    # Optional comparison
    if compare_enabled and option_b_cost and option_b_minutes:
        try:
            b_cost = _parse_positive_float(option_b_cost, "Option B cost")
            b_minutes = _parse_positive_float(option_b_minutes, "Option B minutes")
            b_freq = option_b_frequency if option_b_frequency in frequency_options else frequency
            b_sessions_per_month = sessions_per_month_map.get(b_freq, sessions_per_month)
            b_timeframe_months = timeframe_months
            if mode == "enjoyment":
                b_sessions_total = (b_sessions_per_month * b_timeframe_months) or 1.0
                b_total_hours = b_sessions_total * (b_minutes / 60.0)
                b_cph = b_cost / max(b_total_hours, 0.01)
                winner = "A" if metric_a_primary is not None and metric_a_primary <= b_cph else "B"
                diff = abs((metric_a_primary or 0) - b_cph)
                threshold_sessions = cost / max(b_cph * (minutes_per_occurrence / 60.0) * max(timeframe_months, 0.01), 0.01)
                threshold_per_month = threshold_sessions / max(timeframe_months, 1.0)
                break_even_note = f"At the entered frequencies, {winner} is cheaper by ${diff:,.2f}/hr."
                break_even_note += f" If you use A at least {threshold_per_month:,.2f} times/month, it beats B."
                comparison = {
                    "enabled": True,
                    "mode": mode,
                    "a": {"name": item_name or "Option A", "primary_metric_value": metric_a_primary},
                    "b": {"name": option_b_name or "Option B", "primary_metric_value": b_cph},
                    "winner": winner,
                    "difference_per_hour": diff,
                    "break_even_note": break_even_note,
                }
            else:
                b_hourly = _parse_positive_float(option_b_hourly_raw, "Option B hourly value") if str(option_b_hourly_raw).strip() else _parse_positive_float(hourly_value_raw, "Your hourly value")
                b_value_per_use = (b_minutes / 60.0) * b_hourly
                b_break_even_uses = math.ceil(b_cost / max(b_value_per_use, 0.01))
                winner = "A" if metric_a_break_even is not None and metric_a_break_even <= b_break_even_uses else "B"
                diff = abs((metric_a_break_even or 0) - b_break_even_uses)
                break_even_note = f"{winner} wins with fewer break-even uses (diff {diff})."
                comparison = {
                    "enabled": True,
                    "mode": mode,
                    "a": {"name": item_name or "Option A", "primary_metric_value": metric_a_break_even},
                    "b": {"name": option_b_name or "Option B", "primary_metric_value": b_break_even_uses},
                    "winner": winner,
                    "difference_per_hour": diff,
                    "break_even_note": break_even_note,
                }
        except ValueError:
            comparison = None

    return {
        "output": {
            "share_card": share_card,
            "comparison": comparison,
            "nudge": nudge_data,
            "details": details,
        }
    }
