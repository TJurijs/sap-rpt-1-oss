"""
Shared FastAPI dependencies: Hugging Face authentication, settings injection,
and utility helpers for the playground backend.
"""

from __future__ import annotations

import logging
from typing import Annotated

import os

from fastapi import Depends
from huggingface_hub import login

from playground.backend.config import Settings, configure_logging, get_settings

logger = logging.getLogger(__name__)


def setup_environment() -> Settings:
    """
    Initialize logging and return settings.

    Designed to run once on import, ensuring logging/config is ready
    before request handling starts.
    """

    configure_logging()
    settings = get_settings()
    os.environ.setdefault("HF_HUB_CACHE", str(settings.cache_dir))

    if settings.huggingface_token:
        try:
            login(token=settings.huggingface_token, add_to_git_credential=False)
            logger.info("Authenticated with Hugging Face Hub")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to authenticate with Hugging Face Hub: %s", exc)
            raise
    else:
        logger.warning("HUGGINGFACE_API_KEY not provided; checkpoint downloads will fail.")

    return settings


SETTINGS = setup_environment()


def get_settings_dependency() -> Settings:
    """FastAPI dependency wrapper for settings."""

    return SETTINGS


