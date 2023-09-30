from dataclasses import dataclass, field
from enum import IntEnum

from omegaconf import MISSING


class ConnectorType(IntEnum):
    redis = 0
    prometheus = 1
    druid = 2


@dataclass
class ConnectorConf:
    url: str


@dataclass
class PrometheusConf(ConnectorConf):
    pushgateway: str
    scrape_interval: int = 30


@dataclass
class RedisConf(ConnectorConf):
    port: int
    expiry: int = 300
    master_name: str = "mymaster"
    model_expiry_sec: int = 172800  # 48 hrs
    jitter_secs: int = 30 * 60  # 30 minutes


@dataclass
class Pivot:
    index: str = "timestamp"
    columns: list[str] = field(default_factory=list)
    value: list[str] = field(default_factory=lambda: ["count"])


@dataclass
class DruidFetcherConf:
    datasource: str
    dimensions: list[str] = field(default_factory=list)
    aggregations: dict = field(default_factory=dict)
    group_by: list[str] = field(default_factory=list)
    pivot: Pivot = field(default_factory=lambda: Pivot())
    granularity: str = "minute"

    def __post_init__(self):
        from pydruid.utils.aggregators import doublesum

        if not self.aggregations:
            self.aggregations = {"count": doublesum("count")}


@dataclass
class DruidConf(ConnectorConf):
    """
    Class for configuring Druid connector.

    Args:
        endpoint: Druid endpoint
        delay_hrs: Delay in hours for fetching data from Druid
        fetcher: DruidFetcherConf
    """

    endpoint: str
    delay_hrs: float = 3.0
    fetcher: DruidFetcherConf = MISSING
