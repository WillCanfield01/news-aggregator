# patchpal/install_store.py
from __future__ import annotations
from datetime import datetime
from typing import Optional, Dict, Any
import os, json, pathlib

from .storage import SessionLocal, Installation

# ---- CRUD -------------------------------------------------------------------

def upsert_installation(
    *,
    team_id: str,
    team_name: str,
    bot_token: str,
    bot_user: str,
    scopes: str = "",
    installed_by_user_id: str | None = None,
    enterprise_id: str | None = None,
) -> None:
    """Insert/update the installation for a workspace."""
    with SessionLocal() as db:
        inst = db.get(Installation, team_id)
        now = datetime.utcnow()
        if inst is None:
            inst = Installation(team_id=team_id, installed_at=now)
            db.add(inst)

        inst.team_name = team_name
        inst.bot_token = bot_token
        inst.bot_user  = bot_user
        inst.scopes    = scopes or ""
        inst.installed_by_user_id = installed_by_user_id
        inst.enterprise_id = enterprise_id
        inst.updated_at = now
        db.commit()

def get_bot_token(team_id: str) -> Optional[str]:
    # Avoid SELECT ... WHERE team_id IS NULL
    if not team_id:
        return None
    with SessionLocal() as db:
        inst = db.get(Installation, team_id)
        return inst.bot_token if inst else None

def get_install(team_id: str) -> Optional[Installation]:
    with SessionLocal() as db:
        return db.get(Installation, team_id)

def delete_install(team_id: str) -> None:
    with SessionLocal() as db:
        inst = db.get(Installation, team_id)
        if inst:
            db.delete(inst)
            db.commit()

# ---- One-time migration from file store ------------------------------------

def migrate_file_store_if_present() -> None:
    """
    If a legacy JSON install store exists, import it into the DB once.
    We look at PP_INSTALL_STORE or the old default to keep surprises minimal.
    """
    legacy_path = os.getenv(
        "PP_INSTALL_STORE",
        "/opt/render/project/data/installations.json"
    )
    p = pathlib.Path(legacy_path)
    if not p.exists():
        return

    try:
        data: Dict[str, Dict[str, Any]] = json.loads(p.read_text() or "{}")
    except Exception:
        return

    if not isinstance(data, dict) or not data:
        return

    imported = 0
    with SessionLocal() as db:
        for team_id, rec in data.items():
            if not team_id:
                continue
            if db.get(Installation, team_id):
                continue  # already imported
            inst = Installation(
                team_id=team_id,
                team_name=rec.get("team_name"),
                bot_token=rec.get("bot_token"),
                bot_user=rec.get("bot_user"),
                scopes=rec.get("scopes") or "",
                installed_by_user_id=rec.get("installed_by_user_id"),
            )
            db.add(inst); imported += 1
        if imported:
            db.commit()

    # Mark as migrated so we don't keep re-reading it.
    try:
        p.rename(p.with_suffix(p.suffix + ".migrated"))
    except Exception:
        pass
