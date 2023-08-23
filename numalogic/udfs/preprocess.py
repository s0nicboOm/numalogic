import logging
import os
import time
from dataclasses import replace
from typing import Optional

import orjson
from pynumaflow.function import Datum, Messages, Message
from sklearn.pipeline import make_pipeline

from numalogic.config import PreprocessFactory
from numalogic.registry import LocalLRUCache, RedisRegistry
from numalogic.tools.types import redis_client_t, artifact_t
from numalogic.udfs import NumalogicUDF
from numalogic.udfs._config import StreamConf
from numalogic.udfs.entities import Status, Header
from numalogic.udfs.tools import make_stream_payload, get_df, _load_model

LOCAL_CACHE_TTL = int(os.getenv("LOCAL_CACHE_TTL", "3600"))
LOCAL_CACHE_SIZE = int(os.getenv("LOCAL_CACHE_SIZE", "10000"))

_LOGGER = logging.getLogger(__name__)


class PreprocessUDF(NumalogicUDF):
    """
    Preprocess UDF for Numalogic.

    Args:
        r_client: Redis client
        stream_conf: StreamConf configuration
    """

    def __init__(self, r_client: redis_client_t, stream_conf: Optional[StreamConf] = None):
        super().__init__(is_async=False)
        self.local_cache = LocalLRUCache(cachesize=LOCAL_CACHE_SIZE, ttl=LOCAL_CACHE_TTL)
        self.model_registry = RedisRegistry(client=r_client, cache_registry=self.local_cache)
        self.stream_conf = stream_conf or StreamConf()
        self.preproc_factory = PreprocessFactory()

    def exec(self, keys: list[str], datum: Datum) -> Messages:
        """
        The preprocess function here receives data from the data source.

        Perform preprocess on the input data.

        Args:
        -------
        keys: List of keys
        datum: Datum object

        Returns
        -------
        Messages instance

        """
        _start_time = time.perf_counter()
        messages = Messages()

        # check message sanity
        try:
            data_payload = orjson.loads(datum.value)
            _LOGGER.info("%s - Data payload: %s", data_payload["uuid"], data_payload)
        except (orjson.JSONDecodeError, KeyError) as err:  # catch json decode error only
            _LOGGER.exception("Error while decoding input json: %r", err)
            messages.append(Message.to_drop())
            return messages

        raw_df, timestamps = get_df(data_payload=data_payload, stream_conf=self.stream_conf)

        # Drop message if dataframe shape conditions are not met
        if raw_df.shape[0] < self.stream_conf.window_size or raw_df.shape[1] != len(
            self.stream_conf.metrics
        ):
            _LOGGER.error("Dataframe shape: (%f, %f) error ", raw_df.shape[0], raw_df.shape[1])
            messages.append(Message.to_drop())
            return messages
        # Make StreamPayload object
        payload = make_stream_payload(data_payload, raw_df, timestamps, keys)

        # Check if model will be present in registry
        if any([_conf.stateful for _conf in self.stream_conf.numalogic_conf.preprocess]):
            preproc_artifact = _load_model(
                skeys=keys,
                dkeys=[_conf.name for _conf in self.stream_conf.numalogic_conf.preprocess],
                payload=payload,
                model_registry=self.model_registry,
            )
            if preproc_artifact:
                preproc_clf = preproc_artifact.artifact
                _LOGGER.info(
                    "%s - Loaded model from: %s",
                    payload.uuid,
                    preproc_artifact.extras.get("source"),
                )
            else:
                payload = replace(
                    payload, status=Status.ARTIFACT_NOT_FOUND, header=Header.TRAIN_REQUEST
                )
                return Messages(Message(keys=keys, value=payload.to_json()))
        # Model will not be in registry
        else:
            # Load configuration for the config_id
            _LOGGER.info("%s - Initializing model from config: %s", payload.uuid, payload)
            preproc_clf = self._load_model_from_config(self.stream_conf.numalogic_conf.preprocess)
        try:
            processed_data = self.compute(model=preproc_clf, input_=payload.get_data())
            payload = replace(payload, status=Status.ARTIFACT_FOUND, header=Header.MODEL_INFERENCE)
            _LOGGER.info(
                "%s - Successfully preprocessed, Keys: %s, Metrics: %s, x_scaled: %s",
                payload.uuid,
                keys,
                payload.metrics,
                list(processed_data),
            )
        except RuntimeError:
            _LOGGER.exception(
                "%s - Runtime inference error! Keys: %s, Metric: %s",
                payload.uuid,
                payload.composite_keys,
                payload.metrics,
            )
            payload = replace(payload, status=Status.RUNTIME_ERROR, header=Header.TRAIN_REQUEST)
        messages.append(Message(keys=keys, value=payload.to_json()))
        _LOGGER.debug(
            "%s - Time taken to execute Preprocess: %.4f sec",
            payload.uuid,
            time.perf_counter() - _start_time,
        )
        return messages

    def compute(self, model: artifact_t, input_):
        """
        Perform inference on the input data.

        Args:
            model: Model artifact
            input_: Input data

        Returns
        -------
            Preprocessed array

        Raises
        ------
            RuntimeError: If model forward pass fails
        """
        _start_time = time.perf_counter()
        try:
            x_scaled = model.transform(input_)
        except Exception as err:
            raise RuntimeError("Model transform failed!") from err
        _LOGGER.info("Time taken in preprocessing: %.4f sec", time.perf_counter() - _start_time)
        return x_scaled

    def _load_model_from_config(self, preprocess_cfg):
        preproc_clfs = []
        for _cfg in preprocess_cfg:
            _clf = self.preproc_factory.get_instance(_cfg)
            preproc_clfs.append(_clf)
        return make_pipeline(*preproc_clfs)