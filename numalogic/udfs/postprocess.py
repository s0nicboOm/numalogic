import logging
import os
import time
from dataclasses import replace

import numpy as np
from numpy._typing import NDArray
from orjson import orjson
from pynumaflow.function import Messages, Datum, Message

from numalogic.config import PostprocessFactory
from numalogic.registry import LocalLRUCache, RedisRegistry
from numalogic.tools.types import redis_client_t, artifact_t
from numalogic.udfs import NumalogicUDF
from numalogic.udfs._config import StreamConf
from numalogic.udfs.entities import StreamPayload, Header, Status, TrainerPayload
from numalogic.udfs.tools import _load_model

LOCAL_CACHE_TTL = int(os.getenv("LOCAL_CACHE_TTL", "3600"))
LOCAL_CACHE_SIZE = int(os.getenv("LOCAL_CACHE_SIZE", "10000"))

_LOGGER = logging.getLogger(__name__)


class PostProcessUDF(NumalogicUDF):
    """
    Postprocess UDF for Numalogic.

    Args:
        r_client: Redis client
        stream_conf: StreamConf configuration

    """

    def __init__(self, r_client: redis_client_t, stream_conf: StreamConf = None):
        super().__init__()
        self.model_registry = RedisRegistry(
            client=r_client,
            cache_registry=LocalLRUCache(ttl=LOCAL_CACHE_TTL, cachesize=LOCAL_CACHE_SIZE),
        )
        self.stream_conf = stream_conf or StreamConf()
        self.postproc_factory = PostprocessFactory()

    def exec(self, keys: list[str], datum: Datum) -> Messages:
        """
        The postprocess function here receives data from the previous udf.
        Args:
        -------
        keys: List of keys
        datum: Datum object.

        Returns
        -------
        Messages instance

        """
        _start_time = time.perf_counter()
        messages = Messages()

        # Construct payload object
        payload = StreamPayload(**orjson.loads(datum.value))

        # load configs
        thresh_cfg = self.stream_conf.numalogic_conf.threshold
        postprocess_cfg = self.stream_conf.numalogic_conf.postprocess

        # load artifact
        thresh_artifact = _load_model(
            skeys=keys, dkeys=[thresh_cfg.name], payload=payload, model_registry=self.model_registry
        )
        postproc_clf = self.postproc_factory.get_instance(postprocess_cfg)

        if thresh_artifact is None:
            payload = replace(
                payload, status=Status.ARTIFACT_NOT_FOUND, header=Header.TRAIN_REQUEST
            )

        #  Postprocess payload
        if payload.status in (Status.ARTIFACT_FOUND, Status.ARTIFACT_STALE) and thresh_artifact:
            try:
                processed_data = self.compute(
                    model=thresh_artifact.artifact,
                    input_=payload.get_data(),
                    postproc_clf=postproc_clf,
                )
            except RuntimeError:
                _LOGGER.exception(
                    "%s - Runtime postprocess error! Keys: %s, Metric: %s",
                    payload.uuid,
                    payload.composite_keys,
                    payload.metrics,
                )
                payload = replace(payload, status=Status.RUNTIME_ERROR, header=Header.TRAIN_REQUEST)
            else:
                payload = replace(
                    payload,
                    data=processed_data,
                    header=Header.MODEL_INFERENCE,
                )
                _LOGGER.info(
                    "%s - Successfully post-processed, Keys: %s, Metrics: %s, x_scaled: %s",
                    payload.uuid,
                    keys,
                    payload.metrics,
                    list(processed_data),
                )
                messages.append(Message(keys=keys, value=payload.to_json(), tags=["output"]))

        # Forward payload if a training request is tagged
        if payload.header == Header.TRAIN_REQUEST or payload.status == Status.ARTIFACT_STALE:
            train_payload = TrainerPayload(
                uuid=payload.uuid,
                composite_keys=keys,
                metrics=payload.metrics,
                config_id=payload.config_id,
            )
            messages.append(Message(keys=keys, value=train_payload.to_json(), tags=["train"]))
        _LOGGER.debug(
            "%s -  Time taken in postprocess: %.4f sec",
            payload.uuid,
            time.perf_counter() - _start_time,
        )
        return messages

    def compute(
        self, model: artifact_t, input_: NDArray[float], postproc_clf=None, **_
    ) -> NDArray[float]:
        """
        Compute the postprocess function.

        Args:
        -------
        model: Model instance
        input_: Input data
        kwargs: Additional arguments

        Returns
        -------
        Output data
        """
        _start_time = time.perf_counter()
        try:
            y_score = model.score_samples(input_)
        except Exception as err:
            raise RuntimeError("Threshold model scoring failed") from err
        try:
            win_score = np.mean(y_score, axis=0)
            score = postproc_clf.transform(win_score)
            _LOGGER.debug(
                "Time taken in postprocess compute: %.4f sec", time.perf_counter() - _start_time
            )
        except Exception as err:
            raise RuntimeError("Postprocess failed") from err
        else:
            return score
