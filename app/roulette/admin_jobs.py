from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Optional, Tuple

from flask import current_app

from app.extensions import db
from app.roulette.models import RouletteRegenJob
from app.scripts.generate_timeline_round import ensure_today_round

_runner_started = False
_runner_lock = threading.Lock()
_process_lock = threading.Lock()


def _update_job(job: RouletteRegenJob, status: Optional[str] = None, progress: Optional[int] = None, error: Optional[str] = None):
    if status:
        job.status = status
    if progress is not None:
        job.progress = progress
    if error:
        job.error_message = error[:500]
    db.session.commit()


def _run_job(job_id: int):
    start_ts = time.time()
    job = RouletteRegenJob.query.filter_by(id=job_id).first()
    if not job:
        return
    try:
        job.started_at = datetime.utcnow()
        _update_job(job, status="running", progress=5)

        # Step 1: kickoff
        if time.time() - start_ts > 90:
            raise TimeoutError("time budget exceeded")

        # Step 2: generate today's round (capped)
        ensure_today_round(force=job.force if job.force in (0, 1, 2) else 0)
        job.rounds_generated = 1
        job.puzzles_generated = 3
        _update_job(job, progress=80)

        if time.time() - start_ts > 90:
            raise TimeoutError("time budget exceeded")

        # Step 3: finalize
        job.finished_at = datetime.utcnow()
        _update_job(job, status="done", progress=100)
    except Exception as exc:  # noqa: BLE001
        db.session.rollback()
        job = RouletteRegenJob.query.filter_by(id=job_id).first()
        if not job:
            return
        job.finished_at = datetime.utcnow()
        job.status = "error"
        job.error_message = str(exc)[:500]
        job.progress = min(job.progress or 0, 99)
        db.session.commit()


def _worker(app):
    with app.app_context():
        while True:
            try:
                with _process_lock:
                    job = (
                        db.session.query(RouletteRegenJob)
                        .with_for_update(skip_locked=True)
                        .filter(RouletteRegenJob.status == "queued")
                        .order_by(RouletteRegenJob.created_at.asc())
                        .first()
                    )
                    if not job:
                        db.session.commit()
                        time.sleep(2)
                        continue
                    job.status = "running"
                    job.started_at = datetime.utcnow()
                    job.progress = 5
                    db.session.commit()
                    _run_job(job.id)
            except Exception:  # noqa: BLE001
                db.session.rollback()
            time.sleep(1)


def start_worker(app):
    global _runner_started
    with _runner_lock:
        if _runner_started:
            return
        _runner_started = True
        t = threading.Thread(target=_worker, args=(app,), daemon=True)
        t.start()


def enqueue_job(force: int, requested_ip: Optional[str]) -> RouletteRegenJob:
    job = RouletteRegenJob(
        status="queued",
        progress=0,
        force=force,
        requested_by_ip=requested_ip,
    )
    db.session.add(job)
    db.session.commit()
    return job


def get_running_job() -> Optional[RouletteRegenJob]:
    return (
        RouletteRegenJob.query.filter(RouletteRegenJob.status.in_(["running"]))
        .order_by(RouletteRegenJob.created_at.desc())
        .first()
    )


def job_status_payload(job: RouletteRegenJob) -> dict:
    step = 1
    if job.progress >= 80:
        step = 3
    elif job.progress >= 40:
        step = 2
    return {
        "job_id": job.id,
        "status": job.status,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "message": job.error_message or job.status,
        "progress": {"step": step, "total_steps": 3, "percent": job.progress or 0},
    }
