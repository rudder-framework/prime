"""
Prime Streaming Module

Real-time data streaming and progressive analysis capabilities.
"""

from .analyzers import RealTimeAnalyzer
from .data_sources import (
    get_stream_connector,
    CryptoStreamConnector,
    TurbofanSimulator,
    ChemicalReactorSimulator,
    SystemMetricsConnector,
    SyntheticDataSource,
    DATA_SOURCES,
)

__all__ = [
    'RealTimeAnalyzer',
    'get_stream_connector',
    'CryptoStreamConnector',
    'TurbofanSimulator',
    'ChemicalReactorSimulator',
    'SystemMetricsConnector',
    'SyntheticDataSource',
    'DATA_SOURCES',
]
