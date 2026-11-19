import unittest

import duckdb
import pandas as pd

from fraud import ROOT, ranked, threshold_at_budget


class FeatureTests(unittest.TestCase):
    def test_history_includes_window_start_but_excludes_timestamp_peers(self):
        with duckdb.connect() as db:
            db.execute("""
                CREATE TABLE transactions AS
                SELECT tx_id, ts::timestamp AS tx_time, customer_id,
                       1 AS terminal_id, amount, 0 AS fraud, 0 AS scenario
                FROM (VALUES
                    (1, '2018-01-01 09:00:00', 1, 10.0),
                    (2, '2018-01-01 10:00:00', 1, 20.0),
                    (3, '2018-01-01 10:00:00', 1, 1000.0),
                    (4, '2018-01-01 09:59:59', 2, 500.0),
                    (5, '2018-01-01 10:00:01', 1, 2000.0)
                ) AS example(tx_id, ts, customer_id, amount)
            """)
            features = (ROOT / "sql" / "features.sql").read_text()
            check = """
                SELECT count_1h, count_24h, mean_amount_7d, seconds_since_previous
                FROM features WHERE tx_id IN (2, 3) ORDER BY tx_id
            """
            db.execute(f"CREATE VIEW features AS {features}")
            self.assertEqual(db.execute(check).fetchall(), [(1, 1, 10.0, 3600)] * 2)
            db.execute("DELETE FROM transactions WHERE tx_id = 5")
            self.assertEqual(db.execute(check).fetchall(), [(1, 1, 10.0, 3600)] * 2)


class PolicyTests(unittest.TestCase):
    def test_batch_budget_resets_daily_with_deterministic_ties(self):
        frame = pd.DataFrame({
            "tx_id": [4, 2, 3, 1],
            "tx_time": pd.to_datetime(["2018-01-02", "2018-01-01", "2018-01-02", "2018-01-01"]),
        })
        ordered = ranked(frame, [0.9, 0.8, 0.9, 0.8])
        self.assertEqual(ordered.loc[ordered["rank"].le(1), "tx_id"].tolist(), [1, 3])

    def test_threshold_keeps_score_ties_together(self):
        scores = pd.Series([0.9, 0.8, 0.8, 0.7])
        self.assertEqual(threshold_at_budget(scores, 2), 0.9)
        self.assertEqual(threshold_at_budget(scores, 3), 0.8)
        self.assertEqual(threshold_at_budget(scores, 0), float("inf"))


if __name__ == "__main__":
    unittest.main()
