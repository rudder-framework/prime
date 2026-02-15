"""
Prime Concierge AI
==================

Natural language interface to Prime analysis.
Users ask questions, Concierge generates SQL and returns insights.

Example:
    concierge = Concierge("/path/to/data/")
    answer = concierge.ask("Which entity is healthiest?")
    print(answer)
"""

import duckdb
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import re


@dataclass
class ConciergeResponse:
    """Response from the Concierge."""
    question: str
    answer: str
    sql: str
    data: Optional[List[Dict]] = None
    confidence: str = "high"  # high, medium, low


class Concierge:
    """
    Natural language interface to Prime data.

    Translates questions into SQL queries and returns human-readable answers.
    """

    def __init__(self, data_dir: str):
        """
        Initialize Concierge with path to parquet files.

        Args:
            data_dir: Directory containing physics.parquet, geometry.parquet, etc.
        """
        self.data_dir = Path(data_dir)
        self.conn = duckdb.connect()
        self._load_data()

        # Question patterns and their SQL handlers
        self.patterns = [
            # Health questions
            (r"(healthiest|best|strongest)", self._query_healthiest),
            (r"(sickest|worst|weakest|critical)", self._query_sickest),
            (r"(health|status|condition)\s*(of|for)?\s*(\w+)?", self._query_health),

            # Comparison questions
            (r"compare|difference|versus|vs", self._query_compare),
            (r"rank|order|sort", self._query_rank),

            # Specific metrics
            (r"coherence", self._query_coherence),
            (r"energy", self._query_energy),
            (r"state|diverge|distance", self._query_state),
            (r"prime signal", self._query_prime_signal),

            # Time/trend questions
            (r"(when|first|start).*(fail|degrad|problem|issue)", self._query_first_failure),
            (r"(trend|over time|progress|evolv)", self._query_trends),
            (r"(recover|improv|better)", self._query_recovery),

            # Cause questions
            (r"(why|cause|reason|because)", self._query_causation),
            (r"(what happened|explain)", self._query_explanation),

            # Entity-specific
            (r"(tell me about|describe|summary|summarize)\s*(\w+)?", self._query_entity_summary),

            # Counts/stats
            (r"how many", self._query_counts),
            (r"(average|mean|typical)", self._query_averages),

            # Anomaly questions
            (r"(anomal|unusual|different|outlier|strange)", self._query_anomalies),
        ]

    def _load_data(self):
        """Load parquet files into DuckDB."""
        parquets = {
            'physics': 'physics.parquet',
            'geometry': 'geometry.parquet',
            'primitives': 'primitives.parquet',
        }

        for table, filename in parquets.items():
            path = self.data_dir / filename
            if path.exists():
                self.conn.execute(f"""
                    CREATE OR REPLACE TABLE {table} AS
                    SELECT * FROM read_parquet('{path}')
                """)

    def ask(self, question: str) -> ConciergeResponse:
        """
        Ask a question about the data.

        Args:
            question: Natural language question

        Returns:
            ConciergeResponse with answer, SQL, and data
        """
        question_lower = question.lower().strip()

        # Find matching pattern
        for pattern, handler in self.patterns:
            if re.search(pattern, question_lower):
                return handler(question)

        # Default: general summary
        return self._query_general_summary(question)

    def _execute_sql(self, sql: str) -> List[Dict]:
        """Execute SQL and return results as list of dicts."""
        try:
            result = self.conn.execute(sql).fetchdf()
            return result.to_dict('records')
        except Exception as e:
            return [{"error": str(e)}]

    # =========================================================================
    # QUERY HANDLERS
    # =========================================================================

    def _query_healthiest(self, question: str) -> ConciergeResponse:
        sql = """
        WITH health AS (
            SELECT
                entity_id,
                AVG(coherence) as avg_coherence,
                1.0 / (1 + AVG(state_distance)) as state_score,
                1 - (SUM(CASE WHEN dissipation_rate > 0.01 AND coherence < 0.5
                         AND state_velocity > 0.05 THEN 1 ELSE 0 END) * 1.0 / COUNT(*)) as prime_score
            FROM physics
            GROUP BY entity_id
        )
        SELECT entity_id,
               ROUND(avg_coherence, 3) as coherence,
               ROUND(state_score, 3) as state_score,
               ROUND(prime_score, 3) as prime_score,
               ROUND((avg_coherence * 0.4 + state_score * 0.3 + prime_score * 0.3), 3) as health_score
        FROM health
        ORDER BY health_score DESC
        LIMIT 1
        """
        data = self._execute_sql(sql)

        if data and 'entity_id' in data[0]:
            entity = data[0]['entity_id']
            score = data[0]['health_score']
            answer = f"**Entity {entity}** is the healthiest with a composite health score of {score}.\n\n"
            answer += f"- Coherence: {data[0]['coherence']}\n"
            answer += f"- State score: {data[0]['state_score']}\n"
            answer += f"- Signal score: {data[0]['prime_score']}"
        else:
            answer = "Unable to determine healthiest entity."

        return ConciergeResponse(question=question, answer=answer, sql=sql, data=data)

    def _query_sickest(self, question: str) -> ConciergeResponse:
        sql = """
        WITH health AS (
            SELECT
                entity_id,
                AVG(coherence) as avg_coherence,
                AVG(state_distance) as avg_state_distance,
                MAX(state_distance) as max_state_distance,
                SUM(CASE WHEN dissipation_rate > 0.01 AND coherence < 0.5
                    AND state_velocity > 0.05 THEN 1 ELSE 0 END) as prime_signal_count,
                COUNT(*) as total_obs
            FROM physics
            GROUP BY entity_id
        )
        SELECT entity_id,
               ROUND(avg_coherence, 3) as coherence,
               ROUND(avg_state_distance, 1) as avg_state_dist,
               ROUND(max_state_distance, 1) as max_state_dist,
               prime_signal_count,
               ROUND(prime_signal_count * 100.0 / total_obs, 1) as pct_critical
        FROM health
        ORDER BY max_state_distance DESC, prime_signal_count DESC
        LIMIT 1
        """
        data = self._execute_sql(sql)

        if data and 'entity_id' in data[0]:
            entity = data[0]['entity_id']
            max_dist = data[0]['max_state_dist']
            answer = f"**Entity {entity}** is in the worst condition.\n\n"
            answer += f"- Maximum state divergence: **{max_dist}σ** from baseline\n"
            answer += f"- Average coherence: {data[0]['coherence']}\n"
            answer += f"- Critical signal: {data[0]['pct_critical']}% of time in critical state"
        else:
            answer = "Unable to determine entity in worst condition."

        return ConciergeResponse(question=question, answer=answer, sql=sql, data=data)

    def _query_health(self, question: str) -> ConciergeResponse:
        # Check if specific entity mentioned
        entity_match = re.search(r'(entity|unit|bearing)?\s*(\d+)', question.lower())

        if entity_match:
            entity_id = entity_match.group(2)
            sql = f"""
            SELECT
                entity_id,
                ROUND(AVG(coherence), 3) as avg_coherence,
                ROUND(AVG(state_distance), 1) as avg_state_dist,
                ROUND(MAX(state_distance), 1) as max_state_dist,
                ROUND(AVG(total_energy), 4) as avg_energy,
                SUM(CASE WHEN coherence < 0.4 THEN 1 ELSE 0 END) as decoupled_count,
                SUM(CASE WHEN state_velocity > 0.05 THEN 1 ELSE 0 END) as diverging_count,
                COUNT(*) as total_obs
            FROM physics
            WHERE entity_id = '{entity_id}'
            GROUP BY entity_id
            """
        else:
            sql = """
            SELECT
                entity_id,
                ROUND(AVG(coherence), 3) as avg_coherence,
                ROUND(AVG(state_distance), 1) as avg_state_dist,
                CASE
                    WHEN AVG(coherence) > 0.6 AND AVG(state_distance) < 20 THEN '🟢 HEALTHY'
                    WHEN AVG(coherence) > 0.4 AND AVG(state_distance) < 50 THEN '🟡 DEGRADED'
                    ELSE '🔴 CRITICAL'
                END as status
            FROM physics
            GROUP BY entity_id
            ORDER BY avg_state_dist DESC
            """

        data = self._execute_sql(sql)

        if entity_match and data:
            d = data[0]
            status = "🟢 HEALTHY" if d['avg_coherence'] > 0.6 and d['avg_state_dist'] < 20 else \
                     "🟡 DEGRADED" if d['avg_coherence'] > 0.4 else "🔴 CRITICAL"
            answer = f"**Entity {d['entity_id']}** Status: {status}\n\n"
            answer += f"- Average coherence: {d['avg_coherence']}\n"
            answer += f"- State distance: {d['avg_state_dist']}σ (max: {d['max_state_dist']}σ)\n"
            answer += f"- Decoupled: {d['decoupled_count']}/{d['total_obs']} observations\n"
            answer += f"- Diverging: {d['diverging_count']}/{d['total_obs']} observations"
        else:
            answer = "**Fleet Health Status**\n\n"
            for d in data:
                answer += f"- Entity {d['entity_id']}: {d['status']} (coherence: {d['avg_coherence']}, state: {d['avg_state_dist']}σ)\n"

        return ConciergeResponse(question=question, answer=answer, sql=sql, data=data)

    def _query_coherence(self, question: str) -> ConciergeResponse:
        sql = """
        SELECT
            entity_id,
            ROUND(AVG(coherence), 3) as avg_coherence,
            ROUND(MIN(coherence), 3) as min_coherence,
            ROUND(MAX(coherence), 3) as max_coherence,
            ROUND(AVG(effective_dim), 2) as avg_dimensions,
            CASE
                WHEN AVG(coherence) > 0.7 THEN 'Strongly coupled'
                WHEN AVG(coherence) > 0.4 THEN 'Weakly coupled'
                ELSE 'Decoupled'
            END as coupling_status,
            SUM(CASE WHEN coherence < 0.4 THEN 1 ELSE 0 END) as decoupled_periods
        FROM physics
        GROUP BY entity_id
        ORDER BY avg_coherence DESC
        """
        data = self._execute_sql(sql)

        answer = "**Coherence Analysis**\n\n"
        for d in data:
            answer += f"**Entity {d['entity_id']}**: {d['coupling_status']}\n"
            answer += f"  - Average coherence: {d['avg_coherence']} (range: {d['min_coherence']} - {d['max_coherence']})\n"
            answer += f"  - Effective dimensions: {d['avg_dimensions']}\n"
            answer += f"  - Decoupled periods: {d['decoupled_periods']}\n\n"

        return ConciergeResponse(question=question, answer=answer, sql=sql, data=data)

    def _query_energy(self, question: str) -> ConciergeResponse:
        sql = """
        SELECT
            entity_id,
            ROUND(AVG(total_energy), 4) as avg_energy,
            ROUND(SUM(dissipation_rate), 2) as total_dissipated,
            ROUND(AVG(energy_velocity), 6) as avg_energy_flow,
            CASE
                WHEN AVG(energy_velocity) > 0.001 THEN '↑ Accumulating'
                WHEN AVG(energy_velocity) < -0.001 THEN '↓ Dissipating'
                ELSE '→ Stable'
            END as energy_trend
        FROM physics
        GROUP BY entity_id
        ORDER BY total_dissipated DESC
        """
        data = self._execute_sql(sql)

        answer = "**Energy Analysis**\n\n"
        for d in data:
            answer += f"**Entity {d['entity_id']}**: {d['energy_trend']}\n"
            answer += f"  - Average energy: {d['avg_energy']}\n"
            answer += f"  - Total dissipated: {d['total_dissipated']}\n"
            answer += f"  - Flow rate: {d['avg_energy_flow']}\n\n"

        return ConciergeResponse(question=question, answer=answer, sql=sql, data=data)

    def _query_state(self, question: str) -> ConciergeResponse:
        sql = """
        SELECT
            entity_id,
            ROUND(AVG(state_distance), 1) as avg_distance,
            ROUND(MAX(state_distance), 1) as max_distance,
            ROUND(AVG(state_velocity), 4) as avg_velocity,
            CASE
                WHEN AVG(state_velocity) > 0.05 THEN '↗ Diverging'
                WHEN AVG(state_velocity) < -0.05 THEN '↘ Recovering'
                ELSE '→ Stable'
            END as trajectory,
            SUM(CASE WHEN state_velocity > 0.05 THEN 1 ELSE 0 END) as diverging_count,
            SUM(CASE WHEN state_velocity < -0.05 THEN 1 ELSE 0 END) as recovering_count
        FROM physics
        GROUP BY entity_id
        ORDER BY max_distance DESC
        """
        data = self._execute_sql(sql)

        answer = "**State Analysis**\n\n"
        for d in data:
            answer += f"**Entity {d['entity_id']}**: {d['trajectory']}\n"
            answer += f"  - Distance from baseline: {d['avg_distance']}σ (max: {d['max_distance']}σ)\n"
            answer += f"  - Velocity: {d['avg_velocity']}\n"
            answer += f"  - Diverging: {d['diverging_count']} periods | Recovering: {d['recovering_count']} periods\n\n"

        return ConciergeResponse(question=question, answer=answer, sql=sql, data=data)

    def _query_prime_signal(self, question: str) -> ConciergeResponse:
        sql = """
        SELECT
            entity_id,
            SUM(CASE WHEN dissipation_rate > 0.01 AND coherence < 0.5
                AND state_velocity > 0.05 THEN 1 ELSE 0 END) as signal_count,
            COUNT(*) as total_obs,
            ROUND(SUM(CASE WHEN dissipation_rate > 0.01 AND coherence < 0.5
                AND state_velocity > 0.05 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_critical,
            CASE
                WHEN SUM(CASE WHEN dissipation_rate > 0.01 AND coherence < 0.5
                    AND state_velocity > 0.05 THEN 1 ELSE 0 END) > 0 THEN '⚠️ ACTIVE'
                ELSE '✓ Clear'
            END as status
        FROM physics
        GROUP BY entity_id
        ORDER BY signal_count DESC
        """
        data = self._execute_sql(sql)

        answer = "**Prime Signal Detection**\n\n"
        answer += "_The Prime signal indicates symplectic structure loss: dissipating + decoupling + diverging_\n\n"
        for d in data:
            answer += f"**Entity {d['entity_id']}**: {d['status']}\n"
            answer += f"  - Signal detected: {d['signal_count']} / {d['total_obs']} observations ({d['pct_critical']}%)\n\n"

        return ConciergeResponse(question=question, answer=answer, sql=sql, data=data)

    def _query_first_failure(self, question: str) -> ConciergeResponse:
        sql = """
        WITH first_events AS (
            SELECT
                entity_id,
                MIN(CASE WHEN dissipation_rate > 0.01 THEN I END) as first_dissipation,
                MIN(CASE WHEN coherence < 0.5 THEN I END) as first_decoupling,
                MIN(CASE WHEN state_distance > 10 THEN I END) as first_divergence,
                MIN(CASE WHEN dissipation_rate > 0.01 AND coherence < 0.5
                    AND state_velocity > 0.05 THEN I END) as first_prime_signal
            FROM physics
            GROUP BY entity_id
        )
        SELECT
            entity_id,
            first_dissipation,
            first_decoupling,
            first_divergence,
            first_prime_signal,
            LEAST(COALESCE(first_dissipation, 99999),
                  COALESCE(first_decoupling, 99999),
                  COALESCE(first_divergence, 99999)) as earliest_sign
        FROM first_events
        ORDER BY earliest_sign
        """
        data = self._execute_sql(sql)

        answer = "**First Signs of Failure**\n\n"
        for d in data:
            answer += f"**Entity {d['entity_id']}**:\n"
            answer += f"  - First dissipation: I={d['first_dissipation']}\n"
            answer += f"  - First decoupling: I={d['first_decoupling']}\n"
            answer += f"  - First divergence: I={d['first_divergence']}\n"
            if d['first_prime_signal']:
                answer += f"  - **Prime signal onset: I={d['first_prime_signal']}**\n"
            answer += "\n"

        return ConciergeResponse(question=question, answer=answer, sql=sql, data=data)

    def _query_trends(self, question: str) -> ConciergeResponse:
        sql = """
        WITH windows AS (
            SELECT
                entity_id,
                FLOOR(I / 100) * 100 as window,
                AVG(coherence) as coherence,
                AVG(state_distance) as state_dist,
                AVG(total_energy) as energy
            FROM physics
            GROUP BY entity_id, FLOOR(I / 100) * 100
        )
        SELECT
            entity_id,
            window,
            ROUND(coherence, 3) as coherence,
            ROUND(state_dist, 1) as state_distance,
            ROUND(energy, 4) as energy
        FROM windows
        ORDER BY entity_id, window
        """
        data = self._execute_sql(sql)

        answer = "**Trends Over Time**\n\n"
        current_entity = None
        for d in data:
            if d['entity_id'] != current_entity:
                current_entity = d['entity_id']
                answer += f"\n**Entity {current_entity}**:\n"
            answer += f"  I={int(d['window'])}: C={d['coherence']}, d={d['state_distance']}σ, E={d['energy']}\n"

        return ConciergeResponse(question=question, answer=answer, sql=sql, data=data)

    def _query_recovery(self, question: str) -> ConciergeResponse:
        sql = """
        SELECT
            entity_id,
            SUM(CASE WHEN state_velocity < -0.05 THEN 1 ELSE 0 END) as recovering_periods,
            SUM(CASE WHEN state_velocity > 0.05 THEN 1 ELSE 0 END) as degrading_periods,
            COUNT(*) as total,
            ROUND(SUM(CASE WHEN state_velocity < -0.05 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_recovering,
            CASE
                WHEN SUM(CASE WHEN state_velocity < -0.05 THEN 1 ELSE 0 END) >
                     SUM(CASE WHEN state_velocity > 0.05 THEN 1 ELSE 0 END)
                THEN '↘ Net Recovery'
                ELSE '↗ Net Degradation'
            END as overall_trend
        FROM physics
        GROUP BY entity_id
        ORDER BY pct_recovering DESC
        """
        data = self._execute_sql(sql)

        answer = "**Recovery Analysis**\n\n"
        for d in data:
            answer += f"**Entity {d['entity_id']}**: {d['overall_trend']}\n"
            answer += f"  - Recovering: {d['recovering_periods']} periods ({d['pct_recovering']}%)\n"
            answer += f"  - Degrading: {d['degrading_periods']} periods\n\n"

        return ConciergeResponse(question=question, answer=answer, sql=sql, data=data)

    def _query_causation(self, question: str) -> ConciergeResponse:
        sql = """
        WITH force_analysis AS (
            SELECT
                entity_id,
                SUM(CASE WHEN ABS(energy_velocity) > 0.001
                    AND ABS(coherence_velocity) < 0.001 THEN 1 ELSE 0 END) as exogenous,
                SUM(CASE WHEN ABS(energy_velocity) < 0.001
                    AND ABS(coherence_velocity) > 0.001 THEN 1 ELSE 0 END) as endogenous
            FROM physics
            GROUP BY entity_id
        )
        SELECT
            entity_id,
            exogenous,
            endogenous,
            CASE
                WHEN exogenous > endogenous * 1.5 THEN 'Primarily EXOGENOUS (external forces)'
                WHEN endogenous > exogenous * 1.5 THEN 'Primarily ENDOGENOUS (internal dynamics)'
                ELSE 'MIXED forces'
            END as attribution
        FROM force_analysis
        ORDER BY entity_id
        """
        data = self._execute_sql(sql)

        answer = "**Force Attribution Analysis**\n\n"
        answer += "_Exogenous = external forces moving the system_\n"
        answer += "_Endogenous = internal dynamics deforming the system_\n\n"
        for d in data:
            answer += f"**Entity {d['entity_id']}**: {d['attribution']}\n"
            answer += f"  - Exogenous events: {d['exogenous']}\n"
            answer += f"  - Endogenous events: {d['endogenous']}\n\n"

        return ConciergeResponse(question=question, answer=answer, sql=sql, data=data)

    def _query_explanation(self, question: str) -> ConciergeResponse:
        # Check for specific entity
        entity_match = re.search(r'(\d+)', question)

        if entity_match:
            entity_id = entity_match.group(1)
            return self._query_entity_summary(f"summarize entity {entity_id}")
        else:
            return self._query_general_summary(question)

    def _query_entity_summary(self, question: str) -> ConciergeResponse:
        entity_match = re.search(r'(\d+)', question)

        if entity_match:
            entity_id = entity_match.group(1)
            sql = f"""
            WITH summary AS (
                SELECT
                    entity_id,
                    COUNT(*) as n_obs,
                    ROUND(AVG(coherence), 3) as avg_coherence,
                    ROUND(MIN(coherence), 3) as min_coherence,
                    ROUND(AVG(state_distance), 1) as avg_state,
                    ROUND(MAX(state_distance), 1) as max_state,
                    ROUND(SUM(dissipation_rate), 2) as total_dissipation,
                    SUM(CASE WHEN dissipation_rate > 0.01 AND coherence < 0.5
                        AND state_velocity > 0.05 THEN 1 ELSE 0 END) as prime_count
                FROM physics
                WHERE entity_id = '{entity_id}'
                GROUP BY entity_id
            )
            SELECT * FROM summary
            """
            data = self._execute_sql(sql)

            if data:
                d = data[0]
                answer = f"**Entity {entity_id} Summary**\n\n"
                answer += f"📊 **{d['n_obs']} observations analyzed**\n\n"
                answer += f"**Coherence**: {d['avg_coherence']} average (min: {d['min_coherence']})\n"
                answer += f"**State**: {d['avg_state']}σ average, {d['max_state']}σ maximum divergence\n"
                answer += f"**Energy**: {d['total_dissipation']} total dissipated\n"
                answer += f"**Prime Signal**: {d['prime_count']} critical observations\n\n"

                # Interpretation
                if d['avg_coherence'] > 0.6:
                    answer += "✓ System maintains good coupling\n"
                else:
                    answer += "⚠️ System shows weak coupling / decoupling\n"

                if d['max_state'] > 50:
                    answer += "⚠️ Extreme state divergence detected\n"
                elif d['max_state'] > 20:
                    answer += "⚠️ Significant state divergence\n"

                if d['prime_count'] > 0:
                    answer += f"⚠️ Prime signal active ({d['prime_count']} times)\n"
            else:
                answer = f"No data found for entity {entity_id}"
        else:
            return self._query_general_summary(question)

        return ConciergeResponse(question=question, answer=answer, sql=sql, data=data)

    def _query_compare(self, question: str) -> ConciergeResponse:
        sql = """
        SELECT
            entity_id,
            ROUND(AVG(coherence), 3) as coherence,
            ROUND(AVG(state_distance), 1) as state_dist,
            ROUND(AVG(effective_dim), 2) as dimensions,
            ROUND(SUM(dissipation_rate), 2) as dissipation,
            SUM(CASE WHEN dissipation_rate > 0.01 AND coherence < 0.5
                AND state_velocity > 0.05 THEN 1 ELSE 0 END) as prime_signals
        FROM physics
        GROUP BY entity_id
        ORDER BY entity_id
        """
        data = self._execute_sql(sql)

        answer = "**Entity Comparison**\n\n"
        answer += "| Entity | Coherence | State (σ) | Dimensions | Dissipation | Signal |\n"
        answer += "|--------|-----------|-----------|------------|-------------|--------|\n"
        for d in data:
            answer += f"| {d['entity_id']} | {d['coherence']} | {d['state_dist']} | {d['dimensions']} | {d['dissipation']} | {d['prime_signals']} |\n"

        return ConciergeResponse(question=question, answer=answer, sql=sql, data=data)

    def _query_rank(self, question: str) -> ConciergeResponse:
        return self._query_compare(question)

    def _query_counts(self, question: str) -> ConciergeResponse:
        sql = """
        SELECT
            COUNT(DISTINCT entity_id) as n_entities,
            COUNT(*) as total_observations,
            SUM(CASE WHEN coherence < 0.4 THEN 1 ELSE 0 END) as decoupled_obs,
            SUM(CASE WHEN state_distance > 20 THEN 1 ELSE 0 END) as high_divergence_obs,
            SUM(CASE WHEN dissipation_rate > 0.01 AND coherence < 0.5
                AND state_velocity > 0.05 THEN 1 ELSE 0 END) as prime_signal_obs
        FROM physics
        """
        data = self._execute_sql(sql)

        d = data[0]
        answer = "**Data Summary**\n\n"
        answer += f"- **{d['n_entities']} entities** analyzed\n"
        answer += f"- **{d['total_observations']} total observations**\n"
        answer += f"- {d['decoupled_obs']} decoupled observations\n"
        answer += f"- {d['high_divergence_obs']} high-divergence observations (>20σ)\n"
        answer += f"- {d['prime_signal_obs']} Prime signal observations\n"

        return ConciergeResponse(question=question, answer=answer, sql=sql, data=data)

    def _query_averages(self, question: str) -> ConciergeResponse:
        sql = """
        SELECT
            ROUND(AVG(coherence), 3) as avg_coherence,
            ROUND(AVG(state_distance), 1) as avg_state_distance,
            ROUND(AVG(total_energy), 4) as avg_energy,
            ROUND(AVG(effective_dim), 2) as avg_dimensions,
            ROUND(AVG(eigenvalue_entropy), 3) as avg_entropy
        FROM physics
        """
        data = self._execute_sql(sql)

        d = data[0]
        answer = "**Fleet Averages**\n\n"
        answer += f"- Coherence: {d['avg_coherence']}\n"
        answer += f"- State distance: {d['avg_state_distance']}σ\n"
        answer += f"- Energy: {d['avg_energy']}\n"
        answer += f"- Effective dimensions: {d['avg_dimensions']}\n"
        answer += f"- Eigenvalue entropy: {d['avg_entropy']}\n"

        return ConciergeResponse(question=question, answer=answer, sql=sql, data=data)

    def _query_anomalies(self, question: str) -> ConciergeResponse:
        sql = """
        WITH entity_stats AS (
            SELECT
                entity_id,
                AVG(coherence) as coh,
                AVG(state_distance) as state,
                AVG(effective_dim) as dim
            FROM physics
            GROUP BY entity_id
        ),
        entity_ranked AS (
            SELECT
                entity_id,
                ROUND(coh, 3) as coherence,
                ROUND(PERCENT_RANK() OVER (ORDER BY coh), 2) as coh_pctile,
                ROUND(state, 1) as state_dist,
                ROUND(PERCENT_RANK() OVER (ORDER BY state), 2) as state_pctile,
                ROUND(dim, 2) as dimensions,
                ROUND(PERCENT_RANK() OVER (ORDER BY dim), 2) as dim_pctile
            FROM entity_stats
        )
        SELECT *
        FROM entity_ranked
        ORDER BY GREATEST(ABS(state_pctile - 0.5), ABS(coh_pctile - 0.5), ABS(dim_pctile - 0.5)) DESC
        """
        data = self._execute_sql(sql)

        answer = "**Anomaly Detection**\n\n"
        answer += "_Percentile outside [0.05, 0.95] indicates significant deviation from fleet_\n\n"
        for d in data:
            anomalies = []
            if d['coh_pctile'] < 0.05 or d['coh_pctile'] > 0.95:
                anomalies.append(f"coherence (P{int(d['coh_pctile']*100)})")
            if d['state_pctile'] < 0.05 or d['state_pctile'] > 0.95:
                anomalies.append(f"state (P{int(d['state_pctile']*100)})")
            if d['dim_pctile'] < 0.05 or d['dim_pctile'] > 0.95:
                anomalies.append(f"dimensions (P{int(d['dim_pctile']*100)})")

            if anomalies:
                answer += f"**Entity {d['entity_id']}**: Anomalous in {', '.join(anomalies)}\n"
            else:
                answer += f"Entity {d['entity_id']}: Within normal range\n"

        return ConciergeResponse(question=question, answer=answer, sql=sql, data=data)

    def _query_general_summary(self, question: str) -> ConciergeResponse:
        sql = """
        SELECT
            COUNT(DISTINCT entity_id) as n_entities,
            COUNT(*) as n_observations,
            ROUND(AVG(coherence), 3) as fleet_coherence,
            ROUND(AVG(state_distance), 1) as fleet_state_dist,
            SUM(CASE WHEN dissipation_rate > 0.01 AND coherence < 0.5
                AND state_velocity > 0.05 THEN 1 ELSE 0 END) as total_prime_signals
        FROM physics
        """
        data = self._execute_sql(sql)

        d = data[0]
        answer = "**Fleet Overview**\n\n"
        answer += f"Analyzing **{d['n_entities']} entities** with **{d['n_observations']} observations**\n\n"
        answer += f"- Fleet coherence: {d['fleet_coherence']}\n"
        answer += f"- Fleet state distance: {d['fleet_state_dist']}σ\n"
        answer += f"- Total Prime signals: {d['total_prime_signals']}\n\n"
        answer += "Ask me about specific entities, metrics, or comparisons!"

        return ConciergeResponse(question=question, answer=answer, sql=sql, data=data)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def ask_prime(question: str, data_dir: str = ".") -> str:
    """
    Quick function to ask Prime a question.

    Args:
        question: Natural language question
        data_dir: Path to parquet files

    Returns:
        Answer string
    """
    concierge = Concierge(data_dir)
    response = concierge.ask(question)
    return response.answer


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python concierge.py <data_dir> [question]")
        print("       python concierge.py <data_dir>  # Interactive mode")
        sys.exit(1)

    data_dir = sys.argv[1]
    concierge = Concierge(data_dir)

    if len(sys.argv) > 2:
        # Single question mode
        question = " ".join(sys.argv[2:])
        response = concierge.ask(question)
        print(response.answer)
    else:
        # Interactive mode
        print("Prime Concierge AI")
        print("=" * 50)
        print(f"Data: {data_dir}")
        print("Type 'quit' to exit\n")

        while True:
            try:
                question = input("You: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue

                response = concierge.ask(question)
                print(f"\nPrime: {response.answer}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}\n")

        print("\nGoodbye!")
