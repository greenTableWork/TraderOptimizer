from __future__ import annotations

import threading

import optuna

_CREATE_STUDY_LOCK = threading.Lock()


def create_or_load_study(
    *,
    direction: str,
    study_name: str,
    storage: str,
) -> optuna.Study:
    """Serialize Optuna study creation to avoid PostgreSQL DDL lock races."""
    with _CREATE_STUDY_LOCK:
        return optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )


def is_transient_storage_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    transient_markers = (
        "deadlock detected",
        "could not serialize access",
        "serialization failure",
        "lock timeout",
    )
    return any(marker in message for marker in transient_markers)
