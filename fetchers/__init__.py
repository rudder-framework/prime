"""
RUDDER Fetchers — Download and process data from various sources.

Each fetcher:
1. Downloads raw data from source (NASA, UCI, etc.)
2. Converts to observations.parquet + config.json
3. Writes to data/ directory

Usage:
    from framework.fetchers import cmapss_fetcher

    observations = cmapss_fetcher.fetch(config)
"""
