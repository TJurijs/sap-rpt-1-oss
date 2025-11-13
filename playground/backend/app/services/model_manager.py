"""Manage sap-rpt-1-oss estimator lifecycle and embedding server."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
from huggingface_hub.file_download import hf_hub_download

import sap_rpt_oss.constants as rpt_constants
from sap_rpt_oss import SAP_RPT_OSS_Classifier, SAP_RPT_OSS_Regressor
from sap_rpt_oss.scripts.start_embedding_server import start_embedding_server

from playground.backend.config import Settings, get_settings

logger = logging.getLogger(__name__)


DEFAULT_CHECKPOINT = "2025-11-04_sap-rpt-one-oss.pt"


@dataclass
class ModelStatus:
    checkpoint_path: Optional[Path]
    device: str
    dtype: Optional[str]
    embedding_server_port: int
    embedding_server_started: bool
    cuda_available: bool
    torch_version: str


class ModelManager:
    """Singleton-style manager for classifiers/regressors and embedding server."""

    _instance: Optional["ModelManager"] = None
    _lock = threading.Lock()

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._checkpoint_path: Optional[Path] = None
        self._embedding_started = False
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype: Optional[torch.dtype] = None
        self._init_lock = threading.Lock()
        self._zmq_override_applied = False

    @classmethod
    def instance(cls) -> "ModelManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _ensure_checkpoint(self) -> Path:
        if self._checkpoint_path is None:
            logger.info("Downloading sap-rpt-1-oss checkpoint '%s'", DEFAULT_CHECKPOINT)
            path = hf_hub_download(
                repo_id="SAP/sap-rpt-1-oss",
                filename=DEFAULT_CHECKPOINT,
                token=self.settings.huggingface_token,
                cache_dir=self.settings.cache_dir,
            )
            self._checkpoint_path = Path(path)
            logger.info("Checkpoint ready at %s", self._checkpoint_path)
        return self._checkpoint_path

    def _override_zmq_port(self) -> None:
        if self._zmq_override_applied:
            return
        desired_port = int(self.settings.zmq_port)
        if desired_port != rpt_constants.ZMQ_PORT_DEFAULT:
            logger.info(
                "Overriding sap-rpt-1-oss embedding port from %s to %s",
                rpt_constants.ZMQ_PORT_DEFAULT,
                desired_port,
            )
            rpt_constants.ZMQ_PORT_DEFAULT = desired_port
        self._zmq_override_applied = True

    def _start_embedding_server(self) -> None:
        if not self._embedding_started:
            self._override_zmq_port()
            logger.info("Starting embedding server on port %s", self.settings.zmq_port)
            start_embedding_server(
                sentence_embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                gpu_idx=0 if torch.cuda.is_available() else None,
                zmq_port=self.settings.zmq_port,
            )
            self._embedding_started = True

    def _build_estimator(self, task: Literal["classification", "regression"]):
        self._override_zmq_port()
        checkpoint = str(self._ensure_checkpoint().name)
        kwargs = {
            "checkpoint": checkpoint,
            "max_context_size": 1024,
            "bagging": 2,
            "test_chunk_size": 1000,
        }
        if task == "classification":
            estimator = SAP_RPT_OSS_Classifier(**kwargs)
        else:
            estimator = SAP_RPT_OSS_Regressor(**kwargs)

        self._dtype = getattr(estimator, "dtype", None)
        return estimator

    def get_estimator(self, task: Literal["classification", "regression"]):
        with self._init_lock:
            self._start_embedding_server()
            return self._build_estimator(task)

    def status(self) -> ModelStatus:
        return ModelStatus(
            checkpoint_path=str(self._checkpoint_path) if self._checkpoint_path else None,
            device=str(self._device),
            dtype=str(self._dtype) if self._dtype else None,
            embedding_server_port=self.settings.zmq_port,
            embedding_server_started=self._embedding_started,
            cuda_available=torch.cuda.is_available(),
            torch_version=torch.__version__,
        )


def get_model_manager() -> ModelManager:
    return ModelManager.instance()


