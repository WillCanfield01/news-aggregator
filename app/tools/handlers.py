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
