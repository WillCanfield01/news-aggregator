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


# ---- Social Post Polisher ----


def _extract_json_object(text: str) -> str | None:
    """Best-effort extraction of a top-level JSON object from model output."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1].strip()
    return candidate if candidate.startswith("{") and candidate.endswith("}") else None


def _clamp_list(items: Any, *, max_items: int) -> List[str]:
    if not isinstance(items, list):
        return []
    out: List[str] = []
    for it in items:
        if isinstance(it, str):
            s = it.strip()
            if s:
                out.append(s)
        if len(out) >= max_items:
            break
    return out


def run_social_post_polisher(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw_post = (payload.get("raw_post") or "").strip()
    platform = (payload.get("platform") or "LinkedIn").strip()
    tone = (payload.get("tone") or "Professional").strip()
    length_pref = (payload.get("length") or "Slightly shorter").strip()
    call_to_action = (payload.get("call_to_action") or "No").strip()

    if len(raw_post) < 20:
        raise ValueError("Please paste at least 20 characters so there’s something to polish.")

    platform_options = {"LinkedIn", "X (Twitter)", "Instagram", "Threads"}
    if platform not in platform_options:
        raise ValueError("Platform must be LinkedIn, X (Twitter), Instagram, or Threads.")

    tone_options = {"Professional", "Casual", "Confident", "Friendly"}
    if tone not in tone_options:
        raise ValueError("Tone must be Professional, Casual, Confident, or Friendly.")

    length_options = {"Keep similar length", "Slightly shorter", "Much shorter"}
    if length_pref not in length_options:
        raise ValueError("Length preference must be Keep similar length, Slightly shorter, or Much shorter.")

    cta_options = {"Yes", "No"}
    if call_to_action not in cta_options:
        raise ValueError("Include a call to action must be Yes or No.")

    allow_emojis = platform in {"Instagram", "Threads"}
    allow_hashtags = platform in {"LinkedIn", "Instagram"}
    x_limit_hint = 280 if platform == "X (Twitter)" else None

    prompt = f"""
Rewrite the user's post for {platform} to improve clarity and flow, while preserving their voice.

Hard rules:
- Do NOT add emojis unless platform is Instagram or Threads. (If allowed, max 2 and only if it fits the voice.)
- Do NOT add hashtags unless platform is LinkedIn or Instagram. (If used, max 3 and keep them at the very end.)
- Avoid marketing language and hype. No “game-changer”, “revolutionary”, “unlock”, “leverage”, etc.
- Remove filler. Tighten sentences. Keep it human.
- Break long paragraphs into short readable blocks.
- Do not invent facts. Do not add claims the user did not imply.

User preferences:
- Tone: {tone}
- Length preference: {length_pref}
- Include call to action: {call_to_action} (If Yes, keep it subtle and 1 line max.)
""".strip()

    if x_limit_hint:
        prompt += f"\n- For X, aim for <= {x_limit_hint} characters when possible."

    prompt += "\n\nReturn JSON only (no markdown, no extra text) with this exact shape:\n"
    prompt += """{
  "polished_post": "string",
  "summary": { "changes": ["string", "string", "string"] },
  "alt_versions": { "short": "string or empty", "hook_first": "string or empty" }
}"""
    prompt += "\n\nUser draft:\n" + raw_post

    try:
        result = _client.chat.completions.create(
            model=_DEFAULT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise social post editor. Output must be valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=700,
            temperature=0.35,
            timeout=15,
        )
        content = (result.choices[0].message.content or "").strip()
    except Exception as exc:
        raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    parsed: Dict[str, Any] | None = None
    try:
        parsed = json.loads(content)
    except Exception:
        extracted = _extract_json_object(content)
        if extracted:
            try:
                parsed = json.loads(extracted)
            except Exception:
                parsed = None

    if not isinstance(parsed, dict):
        parsed = {}

    polished_post = (parsed.get("polished_post") or "").strip()
    if not polished_post:
        # Fallback: treat full content as polished post (still structured output)
        polished_post = content

    summary_changes = _clamp_list((parsed.get("summary") or {}).get("changes") if isinstance(parsed.get("summary"), dict) else None, max_items=5)

    alt_versions = parsed.get("alt_versions") if isinstance(parsed.get("alt_versions"), dict) else {}
    alt_short = (alt_versions.get("short") or "").strip() if isinstance(alt_versions, dict) else ""
    alt_hook_first = (alt_versions.get("hook_first") or "").strip() if isinstance(alt_versions, dict) else ""

    # Light deterministic cleanup to reduce "AI-ish" extras
    if not allow_emojis:
        polished_post = re.sub(r"[\U0001F300-\U0001FAFF]", "", polished_post)
        alt_short = re.sub(r"[\U0001F300-\U0001FAFF]", "", alt_short)
        alt_hook_first = re.sub(r"[\U0001F300-\U0001FAFF]", "", alt_hook_first)
    if not allow_hashtags:
        polished_post = re.sub(r"(?:^|\s)#[A-Za-z0-9_]+", "", polished_post).strip()
        alt_short = re.sub(r"(?:^|\s)#[A-Za-z0-9_]+", "", alt_short).strip()
        alt_hook_first = re.sub(r"(?:^|\s)#[A-Za-z0-9_]+", "", alt_hook_first).strip()

    output = {
        "platform": platform,
        "original_length": len(raw_post),
        "polished_length": len(polished_post),
        "summary": {"changes": summary_changes},
        "polished_post": polished_post,
        "alt_versions": {
            "short": alt_short,
            "hook_first": alt_hook_first,
        },
    }
    return {"output": output}


# ---- Grocery List (deterministic, DB persisted via token) ----

GROCERY_CATEGORIES: List[str] = [
    "Produce",
    "Dairy",
    "Meat",
    "Pantry",
    "Frozen",
    "Drinks",
    "Snacks",
    "Household",
    "Other",
]


def categorize_grocery_item(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return "Other"

    def has_any(words: List[str]) -> bool:
        return any(w in t for w in words)

    if has_any(
        [
            "apple",
            "banana",
            "berries",
            "strawberry",
            "blueberry",
            "raspberry",
            "avocado",
            "lettuce",
            "spinach",
            "kale",
            "tomato",
            "onion",
            "garlic",
            "potato",
            "carrot",
            "pepper",
            "cucumber",
            "broccoli",
            "cauliflower",
            "mushroom",
            "orange",
            "lemon",
            "lime",
            "grape",
        ]
    ):
        return "Produce"

    if has_any(["milk", "cheese", "yogurt", "butter", "cream", "egg", "eggs"]):
        return "Dairy"

    if has_any(["chicken", "beef", "pork", "turkey", "bacon", "sausage", "ham", "steak", "salmon", "tuna", "fish", "shrimp"]):
        return "Meat"

    if has_any(["frozen", "ice cream", "pizza", "nugget", "fries", "dumpling"]):
        return "Frozen"

    if has_any(["water", "sparkling", "soda", "coffee", "tea", "juice", "beer", "wine", "seltzer", "energy drink"]):
        return "Drinks"

    if has_any(["chips", "crackers", "cookie", "cookies", "chocolate", "nuts", "granola", "popcorn", "snack"]):
        return "Snacks"

    if has_any(
        [
            "paper towel",
            "paper towels",
            "toilet paper",
            "detergent",
            "dish soap",
            "soap",
            "shampoo",
            "conditioner",
            "toothpaste",
            "trash bag",
            "trash bags",
            "cleaner",
            "bleach",
            "sponge",
        ]
    ):
        return "Household"

    if has_any(
        [
            "rice",
            "pasta",
            "bean",
            "beans",
            "lentil",
            "lentils",
            "flour",
            "sugar",
            "cereal",
            "oat",
            "oats",
            "peanut butter",
            "olive oil",
            "oil",
            "vinegar",
            "spice",
            "spices",
            "sauce",
            "broth",
            "can",
            "canned",
        ]
    ):
        return "Pantry"

    return "Other"


def _normalize_grocery_text(text: Any) -> str:
    s = str(text or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def run_grocery_list_create(payload: Dict[str, Any], *, token: str) -> Dict[str, Any]:
    name = _normalize_grocery_text(payload.get("list_name") or "")
    if len(name) > 60:
        raise ValueError("List name is too long (max 60 characters).")
    if not name:
        name = "Groceries"

    now = datetime.now(timezone.utc).isoformat()
    out_payload = {
        "type": "grocery-list",
        "name": name,
        "items": [],
        "updated_at": now,
    }
    return {"token": token, "payload": out_payload}


def run_grocery_list_get(payload: Dict[str, Any]) -> Dict[str, Any]:
    token = _normalize_grocery_text(payload.get("token") or "")
    if not token:
        raise ValueError("Token is required.")
    return {"token": token}


def run_grocery_list_update(payload: Dict[str, Any], current_payload: Dict[str, Any]) -> Dict[str, Any]:
    token = _normalize_grocery_text(payload.get("token") or "")
    if not token:
        raise ValueError("Token is required.")

    name = _normalize_grocery_text(payload.get("name") or payload.get("list_name") or current_payload.get("name") or "")
    if len(name) > 60:
        raise ValueError("List name is too long (max 60 characters).")
    if not name:
        name = "Groceries"

    incoming_items = payload.get("items")
    if not isinstance(incoming_items, list):
        raise ValueError("Items must be a list.")
    if len(incoming_items) > 300:
        raise ValueError("Too many items (max 300).")

    # Preserve created_at for existing items when possible
    existing_by_id: Dict[str, Any] = {}
    for it in (current_payload.get("items") or []):
        if isinstance(it, dict) and it.get("id"):
            existing_by_id[str(it.get("id"))] = it

    normalized: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    for it in incoming_items:
        if not isinstance(it, dict):
            continue
        item_id = _normalize_grocery_text(it.get("id") or "")
        text = _normalize_grocery_text(it.get("text") or "")
        checked = bool(it.get("checked"))
        if not text:
            continue
        if len(text) > 80:
            raise ValueError(f'Item "{text[:20]}..." is too long (max 80 characters).')
        if not item_id:
            item_id = secrets.token_hex(8)

        created_at = (existing_by_id.get(item_id) or {}).get("created_at") or now
        category = categorize_grocery_item(text)
        normalized.append(
            {
                "id": item_id,
                "text": text,
                "category": category,
                "checked": checked,
                "created_at": created_at,
            }
        )

    out_payload = dict(current_payload or {})
    out_payload["type"] = "grocery-list"
    out_payload["name"] = name
    out_payload["items"] = normalized
    out_payload["updated_at"] = now
    return {"token": token, "payload": out_payload}


# ---- Countdown (share payload only; math is client-side) ----

COUNTDOWN_TIMEZONES = {"Local", "America/Boise", "UTC"}


def _parse_iso_date_yyyy_mm_dd(raw: Any, *, label: str) -> str:
    s = _normalize_grocery_text(raw)
    if not s:
        raise ValueError(f"{label} is required.")
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        raise ValueError(f"{label} must be in YYYY-MM-DD format.")
    return s


def run_countdown_create_share(payload: Dict[str, Any], *, token: str) -> Dict[str, Any]:
    name = _normalize_grocery_text(payload.get("event_name") or payload.get("name") or "")
    if not name:
        raise ValueError("Event name is required.")
    if len(name) > 60:
        raise ValueError("Event name is too long (max 60 characters).")

    date = _parse_iso_date_yyyy_mm_dd(payload.get("event_date") or payload.get("date"), label="Event date")
    tz = _normalize_grocery_text(payload.get("timezone") or "Local") or "Local"
    if tz not in COUNTDOWN_TIMEZONES:
        raise ValueError("Timezone must be Local, America/Boise, or UTC.")

    now = datetime.now(timezone.utc).isoformat()
    out_payload = {
        "type": "countdown",
        "name": name,
        "date": date,
        "timezone": tz,
        "created_at": now,
        "updated_at": now,
    }
    return {"token": token, "payload": out_payload}


def run_countdown_get(payload: Dict[str, Any]) -> Dict[str, Any]:
    token = _normalize_grocery_text(payload.get("token") or "")
    if not token:
        raise ValueError("Token is required.")
    return {"token": token}


# ---- Worth It Calculator ----


def _format_currency(val: float) -> str:
    return f"${val:,.2f}"


def _format_hours(val: float) -> str:
    return f"{val:,.2f}".rstrip("0").rstrip(".")


def run_worth_it(payload: Dict[str, Any]) -> Dict[str, Any]:
    item_name = (payload.get("item_name") or "").strip()
    mode = (payload.get("mode") or "enjoyment").strip().lower()
    frequency = (payload.get("frequency") or "One-time").strip()
    notes = (payload.get("notes") or "").strip()

    cost_raw = payload.get("cost") or payload.get("cost_amount") or ""
    uses_raw = payload.get("uses_per_frequency") or ""
    timeframe_raw = payload.get("timeframe_months") or payload.get("timeframe") or ""

    minutes_raw = payload.get("minutes_per_use") or ""
    # Back-compat
    if not str(minutes_raw).strip():
        minutes_raw = payload.get("minutes_per_occurrence") or payload.get("minutes_saved_per_use") or ""

    target_value_raw = payload.get("target_value_per_hour") or payload.get("hourly_value") or ""

    compare_enabled = (payload.get("compare_enabled") or "").lower() in {"yes", "true", "1", "on"}
    option_b_name = (payload.get("option_b_name") or "").strip()
    option_b_cost_raw = payload.get("option_b_cost") or ""
    option_b_minutes_raw = payload.get("option_b_minutes") or ""
    option_b_frequency = (payload.get("option_b_frequency") or "").strip()
    option_b_target_value_raw = payload.get("option_b_hourly_value") or ""

    nudge_enabled = (payload.get("nudge_enabled") or "").lower() in {"yes", "true", "1", "on"}
    return_window_raw = payload.get("return_window_days") or ""
    quit_chance_raw = payload.get("quit_chance_percent") or payload.get("chance_quit_pct") or ""
    follow_through_raw = payload.get("follow_through_percent") or payload.get("probability_use_pct") or ""

    def _parse_money(value: Any, label: str) -> float:
        raw = str(value or "").strip()
        if not raw:
            raise ValueError(f"{label} is required.")
        clean = raw.replace(",", "")
        clean = clean.replace("$", "")
        try:
            amount = float(clean)
        except Exception:
            raise ValueError(f"{label} must be a number (e.g., 25 or 25.99).")
        if amount <= 0:
            raise ValueError(f"{label} must be greater than zero.")
        return amount

    def _parse_positive_int(value: Any, label: str) -> int:
        raw = str(value or "").strip()
        if not raw:
            raise ValueError(f"{label} is required.")
        clean = raw.replace(",", "")
        try:
            as_float = float(clean)
        except Exception:
            raise ValueError(f"{label} must be an integer.")
        as_int = int(as_float)
        if abs(as_float - as_int) > 1e-9:
            raise ValueError(f"{label} must be an integer.")
        if as_int < 1:
            raise ValueError(f"{label} must be at least 1.")
        return as_int

    def _parse_minutes(value: Any, label: str) -> float:
        raw = str(value or "").strip().lower()
        if not raw:
            raise ValueError(f"{label} is required.")
        match = re.match(r"^(\d+(?:\.\d+)?)(?:\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours))?$", raw)
        if not match:
            raise ValueError(f"{label} must be a number (minutes) like 60, or hours like 1.5h.")
        qty = float(match.group(1))
        if qty <= 0:
            raise ValueError(f"{label} must be greater than zero.")
        suffix = raw[len(match.group(1)) :].strip()
        if suffix.startswith(("h", "hr", "hour")):
            return qty * 60.0
        return qty

    def _parse_optional_positive_float(value: Any) -> float | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        clean = raw.replace(",", "").replace("$", "")
        try:
            num = float(clean)
        except Exception:
            raise ValueError("Value per hour must be a number.")
        if num <= 0:
            raise ValueError("Value per hour must be greater than zero.")
        return num

    def _parse_timeframe_months(value: Any) -> int:
        raw = str(value or "").strip().lower()
        if not raw:
            return 12
        m = re.match(r"^(\d+)(?:\s*(months?|mo|m))?$", raw)
        if m:
            months = int(m.group(1))
            if months < 1:
                raise ValueError("Timeframe must be at least 1 month.")
            return months
        y = re.match(r"^(\d+)(?:\s*(years?|yr|y))$", raw)
        if y:
            years = int(y.group(1))
            if years < 1:
                raise ValueError("Timeframe must be at least 1 year.")
            return years * 12
        raise ValueError("Timeframe must be a number of months (e.g., 12) or years (e.g., 1y).")

    if mode not in {"enjoyment", "time_saved"}:
        raise ValueError('Mode must be "enjoyment" or "time_saved".')

    frequency_options = {"One-time", "Daily", "Weekly", "Biweekly", "Monthly", "Yearly"}
    if frequency not in frequency_options:
        raise ValueError("Frequency must be One-time, Daily, Weekly, Biweekly, Monthly, or Yearly.")

    cost = _parse_money(cost_raw, "Cost")
    timeframe_months = _parse_timeframe_months(timeframe_raw)
    uses_per_frequency = _parse_positive_int(uses_raw, "Uses per frequency")
    minutes_per_use = _parse_minutes(
        minutes_raw,
        "Minutes per use" if mode == "enjoyment" else "Minutes saved per use",
    )
    target_value_per_hour = _parse_optional_positive_float(target_value_raw) or 25.0

    days_per_month = 30.4375
    weeks_per_month = 4.34524
    months_per_year = 12.0

    timeframe_years = timeframe_months / months_per_year
    weeks_in_timeframe = timeframe_months * weeks_per_month
    days_in_timeframe = timeframe_months * days_per_month

    if frequency == "Daily":
        periods = days_in_timeframe
    elif frequency == "Weekly":
        periods = weeks_in_timeframe
    elif frequency == "Biweekly":
        periods = weeks_in_timeframe / 2.0
    elif frequency == "Monthly":
        periods = float(timeframe_months)
    elif frequency == "Yearly":
        periods = timeframe_years
    else:  # One-time
        periods = 1.0

    sessions_total = uses_per_frequency * periods
    total_minutes = minutes_per_use * sessions_total
    total_hours = total_minutes / 60.0

    total_cost = cost if frequency == "One-time" else cost * periods
    cost_per_hour = total_cost / max(total_hours, 0.01)

    required_hours = total_cost / max(target_value_per_hour, 0.01)
    hours_per_use = minutes_per_use / 60.0
    required_sessions = required_hours / max(hours_per_use, 0.01)
    sessions_per_month_needed = required_sessions / max(float(timeframe_months), 0.01)
    sessions_per_week_needed = required_sessions / max(weeks_in_timeframe, 0.01)

    friendly_break_even = ""
    if sessions_per_week_needed > 0 and sessions_per_week_needed < 1:
        every_weeks = 1.0 / sessions_per_week_needed
        friendly_break_even = f"≈ 1 use every {every_weeks:,.2f} weeks"
    elif sessions_per_week_needed >= 1:
        friendly_break_even = f"≈ {sessions_per_week_needed:,.2f} uses/week"

    adjusted = None
    verdict_label = "Buy now" if cost_per_hour <= target_value_per_hour else "Skip for now"
    verdict_reason = (
        f"Base cost per hour {_format_currency(cost_per_hour)}/hr vs your value {_format_currency(target_value_per_hour)}/hr."
    )

    if nudge_enabled:
        def _parse_optional_int_clamped(value: Any, label: str, *, min_value: int, max_value: int) -> int:
            raw = str(value or "").strip()
            if not raw:
                return 0
            try:
                as_float = float(raw.replace(",", "").strip())
            except Exception:
                raise ValueError(f"{label} must be a number.")
            as_int = int(as_float)
            if abs(as_float - as_int) > 1e-9:
                raise ValueError(f"{label} must be a whole number.")
            return min(max(as_int, min_value), max_value)

        def _clamp_pct(val: Any, label: str) -> float:
            raw = str(val or "").strip()
            if not raw:
                return 0.0
            try:
                pct = float(raw.replace(",", "").strip())
            except Exception:
                raise ValueError(f"{label} must be a number between 0 and 100.")
            pct = min(max(pct, 0.0), 100.0)
            return pct

        p_use_pct = _clamp_pct(follow_through_raw, "Follow-through percent") if str(follow_through_raw).strip() else 100.0
        p_quit_pct = _clamp_pct(quit_chance_raw, "Quit chance percent") if str(quit_chance_raw).strip() else 0.0
        p_use = p_use_pct / 100.0
        p_quit = p_quit_pct / 100.0
        multiplier = p_use * (1.0 - p_quit)
        multiplier = min(max(multiplier, 0.05), 1.0)
        adjusted_hours = total_hours * multiplier
        adjusted_cost_per_hour = total_cost / max(adjusted_hours, 0.01)
        adjusted = {
            "adjusted_cost_per_hour": adjusted_cost_per_hour,
            "adjusted_display": f"{_format_currency(adjusted_cost_per_hour)}/hr",
            "probability_use_pct": round(p_use_pct, 2),
            "chance_quit_pct": round(p_quit_pct, 2),
            "expected_usage_multiplier": round(multiplier, 3),
            "return_window_days": _parse_optional_int_clamped(
                return_window_raw,
                "Return window days",
                min_value=0,
                max_value=365,
            ),
        }
        verdict_label = "Buy now" if adjusted_cost_per_hour <= target_value_per_hour else "Skip for now"
        verdict_reason = (
            f"Adjusted cost per hour {adjusted['adjusted_display']} vs your value {_format_currency(target_value_per_hour)}/hr."
        )

    primary_label = "Cost per hour" if mode == "enjoyment" else "Cost per hour saved"
    primary_display = f"{_format_currency(cost_per_hour)}/hr"
    primary_value = cost_per_hour
    if adjusted:
        primary_display = adjusted["adjusted_display"]
        primary_value = adjusted["adjusted_cost_per_hour"]

    output: Dict[str, Any] = {
        "title": item_name or "Is this worth it?",
        "verdict": {"label": verdict_label, "reason": verdict_reason},
        "primary": {
            "metric_label": primary_label,
            "metric_value": primary_value,
            "metric_display": primary_display,
        },
        "break_even": {
            "per_week": round(sessions_per_week_needed, 4),
            "per_month": round(sessions_per_month_needed, 4),
            "friendly": friendly_break_even,
        },
        "totals": {
            "total_cost": round(total_cost, 6),
            "total_hours": round(total_hours, 6),
            "sessions_total": round(sessions_total, 6),
            "periods": round(periods, 6),
            "frequency": frequency,
            "timeframe_months": timeframe_months,
        },
        "expected": adjusted,
        "compare": None,
    }

    if mode == "time_saved":
        value_per_use = target_value_per_hour * hours_per_use
        break_even_uses = int(math.ceil(total_cost / max(value_per_use, 0.01)))
        break_even_uses_per_frequency = (break_even_uses / periods) if periods > 0 else None
        output["time_saved"] = {
            "value_per_use": value_per_use,
            "value_per_use_display": _format_currency(value_per_use),
            "break_even_uses": break_even_uses,
            "break_even_uses_per_frequency": round(break_even_uses_per_frequency, 4) if break_even_uses_per_frequency else None,
        }

    if notes:
        output["notes"] = notes

    # Optional compare (keep deterministic, assume same uses_per_frequency/timeframe for B)
    if compare_enabled and str(option_b_cost_raw).strip() and str(option_b_minutes_raw).strip():
        try:
            b_cost = _parse_money(option_b_cost_raw, "Option B cost")
            b_minutes = _parse_minutes(option_b_minutes_raw, "Option B minutes per use")
            b_freq = option_b_frequency if option_b_frequency in frequency_options else frequency
            if b_freq == "Daily":
                b_periods = days_in_timeframe
            elif b_freq == "Weekly":
                b_periods = weeks_in_timeframe
            elif b_freq == "Biweekly":
                b_periods = weeks_in_timeframe / 2.0
            elif b_freq == "Monthly":
                b_periods = float(timeframe_months)
            elif b_freq == "Yearly":
                b_periods = timeframe_years
            else:
                b_periods = 1.0

            b_sessions_total = uses_per_frequency * b_periods
            b_total_cost = b_cost if b_freq == "One-time" else b_cost * b_periods
            b_total_hours = (b_minutes * b_sessions_total) / 60.0
            b_cph = b_total_cost / max(b_total_hours, 0.01)

            a_name = item_name or "Option A"
            b_name = option_b_name or "Option B"
            winner = "A" if cost_per_hour <= b_cph else "B"
            diff = abs(cost_per_hour - b_cph)
            output["compare"] = {
                "enabled": True,
                "mode": mode,
                "a": {"name": a_name, "cost_per_hour": cost_per_hour, "display": f"{_format_currency(cost_per_hour)}/hr"},
                "b": {"name": b_name, "cost_per_hour": b_cph, "display": f"{_format_currency(b_cph)}/hr"},
                "winner": winner,
                "difference_per_hour": diff,
                "note": f"Assumes the same uses-per-frequency for A and B ({uses_per_frequency}).",
            }
        except ValueError:
            output["compare"] = None

    return {"output": output}
