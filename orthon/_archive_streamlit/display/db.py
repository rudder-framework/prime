"""
ORTHON Results Database
=======================

DuckDB connection to PRISM parquet outputs.
SQL is the query interface. No guessing, just queries.
"""

import duckdb
from pathlib import Path
from typing import List, Tuple, Optional, Union
import pandas as pd


class ResultsDB:
    """DuckDB connection to PRISM parquets"""

    # Standard PRISM output parquets
    PARQUETS = {
        "vector": "vector.parquet",
        "geometry": "geometry.parquet",
        "dynamics": "dynamics.parquet",
        "state": "state.parquet",
        "physics": "physics.parquet",
        "fields": "fields.parquet",
        "systems": "systems.parquet",
    }

    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        self.con = duckdb.connect(":memory:")
        self._available_tables: List[str] = []
        self._load_parquets()

    def _load_parquets(self):
        """Create views for each parquet that exists"""
        for name, filename in self.PARQUETS.items():
            path = self.results_path / filename
            if path.exists():
                try:
                    self.con.execute(f"""
                        CREATE VIEW {name} AS
                        SELECT * FROM read_parquet('{path}')
                    """)
                    self._available_tables.append(name)
                except Exception as e:
                    print(f"Warning: Could not load {name}: {e}")

    def query(self, sql: str) -> Union[pd.DataFrame, Tuple[None, str]]:
        """
        Run SQL query, return DataFrame.

        Returns:
            DataFrame on success, or (None, error_message) on failure
        """
        try:
            return self.con.execute(sql).df()
        except Exception as e:
            return None, str(e)

    def tables(self) -> List[str]:
        """List available tables"""
        return self._available_tables.copy()

    def schema(self, table: str) -> List[Tuple[str, str]]:
        """Get columns and types for a table"""
        try:
            result = self.con.execute(f"DESCRIBE {table}").fetchall()
            return [(r[0], r[1]) for r in result]
        except:
            return []

    def column_names(self, table: str) -> List[str]:
        """Get just column names for a table"""
        return [col for col, dtype in self.schema(table)]

    def entities(self) -> List[str]:
        """Get unique entities across all tables"""
        for table in ['vector', 'dynamics', 'geometry']:
            if table in self._available_tables:
                try:
                    result = self.con.execute(f"""
                        SELECT DISTINCT entity_id
                        FROM {table}
                        ORDER BY entity_id
                    """).fetchall()
                    return [r[0] for r in result]
                except:
                    continue
        return []

    def signals(self) -> List[str]:
        """Get unique signals"""
        if 'vector' in self._available_tables:
            try:
                result = self.con.execute("""
                    SELECT DISTINCT signal_id
                    FROM vector
                    ORDER BY signal_id
                """).fetchall()
                return [r[0] for r in result]
            except:
                pass
        return []

    def windows(self) -> List[int]:
        """Get unique window indices"""
        for table in ['vector', 'dynamics', 'geometry']:
            if table in self._available_tables:
                try:
                    result = self.con.execute(f"""
                        SELECT DISTINCT window
                        FROM {table}
                        ORDER BY window
                    """).fetchall()
                    return [r[0] for r in result]
                except:
                    continue
        return []

    def row_count(self, table: str) -> int:
        """Get row count for a table"""
        try:
            result = self.con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            return result[0] if result else 0
        except:
            return 0

    def close(self):
        """Close connection"""
        self.con.close()
