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
    mode = (payload.get("mode") or payload.get("mode_selection") or "Time saved").strip() or "Time saved"
    cost_raw = payload.get("cost") or payload.get("cost_amount") or ""
    frequency = (payload.get("frequency") or "One-time").strip() or "One-time"
    minutes_saved_raw = payload.get("minutes_saved_per_occurrence") or payload.get("minutes") or ""
    minutes_enjoyment_raw = payload.get("minutes_of_enjoyment_per_occurrence") or ""
    hours_legacy = payload.get("hours")
    occurrences_raw = payload.get("occurrences_per_frequency") or ""
    timeframe_raw = payload.get("timeframe") or ""
    hourly_value_raw = payload.get("hourly_value") or ""
    notes = (payload.get("notes") or "").strip()

    cost = _parse_money(cost_raw)

    def _parse_positive_float(value: Any, label: str) -> float:
        try:
            num = float(str(value).strip())
        except Exception:
            raise ValueError(f"{label} must be a number.")
        if num <= 0:
            raise ValueError(f"{label} must be greater than zero.")
        return num

    if mode not in {"Time saved", "Enjoyment"}:
        raise ValueError("Mode must be Time saved or Enjoyment.")

    if not occurrences_raw:
        raise ValueError('"occurrences_per_frequency" is required.')
    occurrences_per_frequency = _parse_positive_float(occurrences_raw, "Occurrences per period")

    minutes_per_occurrence_raw = minutes_saved_raw if mode == "Time saved" else minutes_enjoyment_raw
    if not minutes_per_occurrence_raw and hours_legacy:
        try:
            minutes_per_occurrence_raw = float(hours_legacy) * 60
        except Exception:
            minutes_per_occurrence_raw = ""
    if not minutes_per_occurrence_raw:
        raise ValueError("Minutes per occurrence is required.")
    minutes_per_occurrence = _parse_positive_float(minutes_per_occurrence_raw, "Minutes per occurrence")

    def _parse_timeframe(raw: str, freq: str) -> tuple[float, str]:
        text = (raw or "").strip().lower()
        if not text:
            if freq == "Weekly":
                return 4.0, "weeks"
            if freq == "Monthly":
                return 1.0, "months"
            if freq == "Yearly":
                return 1.0, "years"
            return 1.0, "occurrence"
        match = re.match(r"^(\d+(?:\.\d+)?)(\s*(weeks?|w|months?|m|years?|y))$", text)
        if not match:
            raise ValueError("Timeframe must be like '8 weeks', '12 months', or '1 year'.")
        amount = float(match.group(1))
        unit_raw = match.group(2).strip()
        if amount <= 0:
            raise ValueError("Timeframe must be greater than zero.")
        if unit_raw.startswith("w"):
            return amount, "weeks"
        if unit_raw.startswith("m"):
            return amount, "months"
        if unit_raw.startswith("y"):
            return amount, "years"
        raise ValueError("Timeframe must specify weeks, months, or years.")

    timeframe_amount, timeframe_unit = _parse_timeframe(timeframe_raw, frequency)

    freq = frequency
    if freq not in {"One-time", "Weekly", "Monthly", "Yearly"}:
        raise ValueError("Frequency must be One-time, Weekly, Monthly, or Yearly.")

    periods = 1.0
    if freq == "Weekly":
        if timeframe_unit == "weeks":
            periods = timeframe_amount
        elif timeframe_unit == "months":
            periods = timeframe_amount * 4.0
        elif timeframe_unit == "years":
            periods = timeframe_amount * 52.0
    elif freq == "Monthly":
        if timeframe_unit == "months":
            periods = timeframe_amount
        elif timeframe_unit == "weeks":
            periods = timeframe_amount / 4.0
        elif timeframe_unit == "years":
            periods = timeframe_amount * 12.0
    elif freq == "Yearly":
        if timeframe_unit == "years":
            periods = timeframe_amount
        elif timeframe_unit == "months":
            periods = timeframe_amount / 12.0
        elif timeframe_unit == "weeks":
            periods = timeframe_amount / 52.0
    else:
        periods = 1.0

    occurrences_total = occurrences_per_frequency * periods
    if occurrences_total <= 0:
        raise ValueError("Occurrences over timeframe must be greater than zero.")

    hours_per_occurrence = minutes_per_occurrence / 60.0
    total_hours = hours_per_occurrence * occurrences_total
    if total_hours <= 0:
        raise ValueError("Total hours must be greater than zero.")

    total_minutes = minutes_per_occurrence * occurrences_total
    total_cost = cost * periods
    cost_per_occurrence = total_cost / occurrences_total
    effective_cph = total_cost / total_hours
    minutes_per_dollar = total_minutes / total_cost if total_cost > 0 else 0

    hourly_value_val = None
    if str(hourly_value_raw).strip():
        hourly_value_val = _parse_money(hourly_value_raw)

    def _format_occurrence_label(val: float, unit: str) -> str:
        if unit == "occurrence":
            return "one-time"
        if unit == "weeks":
            return f"{_format_hours(val)} week(s)"
        if unit == "months":
            return f"{_format_hours(val)} month(s)"
        if unit == "years":
            return f"{_format_hours(val)} year(s)"
        return _format_hours(val)

    summary_timeframe = _format_occurrence_label(timeframe_amount, timeframe_unit)

    verdict_line = f"Break-even hourly value: {_format_currency(effective_cph)} per hour."
    extra_benchmark = ""
    if hourly_value_val is not None:
        if mode == "Time saved":
            net_value = (hourly_value_val * total_hours) - total_cost
            verdict_line = "Worth it" if net_value >= 0 else "Not worth it"
            verdict_line += f" at your time value of {_format_currency(hourly_value_val)}/hr (breakeven {_format_currency(effective_cph)}/hr)."
        else:
            verdict_line = f"Good value if you're happy paying {_format_currency(effective_cph)} per hour of enjoyment."
            extra_benchmark = f" Your benchmark: {_format_currency(hourly_value_val)}/hr."

    lines: List[str] = []
    lines.append("Summary")
    lines.append(f"Item: {item_name or 'N/A'}")
    lines.append(f"Mode: {mode}")
    lines.append(f"Frequency: {frequency}")
    lines.append(f"Timeframe: {summary_timeframe}")
    lines.append("")
    lines.append("Inputs")
    lines.append(f"Cost: {_format_currency(cost)}")
    lines.append(f"Minutes per occurrence: {_format_hours(minutes_per_occurrence)}")
    lines.append(f"Occurrences per period: {_format_hours(occurrences_per_frequency)}")
    if hourly_value_val is not None:
        lines.append(f"Optional hourly value: {_format_currency(hourly_value_val)}")
    if notes:
        lines.append(f"Assumptions: {notes}")
    lines.append("")
    lines.append("Results")
    lines.append(f"Occurrences total: {_format_hours(occurrences_total)}")
    lines.append(f"Total cost: {_format_currency(total_cost)}")
    lines.append(f"Total hours: {_format_hours(total_hours)}")
    lines.append(f"Cost per occurrence: {_format_currency(cost_per_occurrence)}")
    lines.append(f"Primary metric — Effective cost per hour: {_format_currency(effective_cph)}")
    lines.append(f"Minutes per dollar: {minutes_per_dollar:,.2f}")
    lines.append("")
    lines.append("Verdict")
    lines.append(f"{verdict_line}{extra_benchmark}")
    lines.append("• Assumes usage is consistent over the stated timeframe.")
    if mode == "Time saved":
        lines.append("• Treats minutes as time saved.")
    else:
        lines.append("• Treats minutes as enjoyment time.")
    lines.append("• Linear scaling of cost and minutes across the timeframe.")

    return {"output": "\n".join(lines)}
