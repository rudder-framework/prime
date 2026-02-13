"""Moved to framework.ingest.data_reader. This re-export preserves imports."""
from framework.ingest.data_reader import DataReader, DataProfile, main

__all__ = ['DataReader', 'DataProfile', 'main']
