import unittest

from tools.parse_window import aggregate_summaries


def summary(rank, total_ns, count=4):
    return {
        "source": {"rank": rank, "segment": 3},
        "parser": {"parse_seconds": 0.1 + rank * 0.01},
        "coverage": {
            "matched_device_tasks": count,
            "named_device_tasks": count,
            "compute_name_coverage": 1.0,
        },
        "steps": [{"global_step": 13, "index_id": 11, "duration_ns": total_ns}],
        "kernels": [
            {
                "name": "kernel_a",
                "name_source": "node_api",
                "host_task_type": "KERNEL_AICORE",
                "stream_id": 2,
                "device_task_type": "AI_CORE",
                "count": count,
                "total_ns": total_ns,
                "p50_ns": total_ns / count,
                "p95_ns": total_ns / count,
                "p99_ns": total_ns / count,
                "max_ns": total_ns / count,
            }
        ],
    }


class ParseWindowTest(unittest.TestCase):
    def test_explicit_non_contiguous_rank_set(self):
        result = aggregate_summaries(
            [summary(2, 100.0), summary(5, 101.0)],
            expected_rank_ids=[2, 5],
        )
        self.assertTrue(result["complete"])
        self.assertEqual(result["expected_rank_ids"], [2, 5])

    def test_aggregate_detects_rank_outlier(self):
        result = aggregate_summaries(
            [summary(0, 100.0), summary(1, 102.0), summary(2, 200.0)],
            ratio_threshold=1.2,
            expected_ranks=3,
            wall_seconds=0.3,
        )
        self.assertTrue(result["complete"])
        self.assertEqual(result["kernel_group_count"], 1)
        self.assertEqual(result["kernel_anomaly_count"], 1)
        self.assertEqual(result["kernel_comparisons"][0]["outlier_ranks"], [2])
        self.assertEqual(result["step_comparisons"][0]["outlier_ranks"], [2])
        self.assertEqual(result["step_comparisons"][0]["global_step"], 13)

    def test_aggregate_detects_missing_rank_and_count_mismatch(self):
        result = aggregate_summaries(
            [summary(0, 100.0), summary(2, 100.0, count=5)],
            expected_ranks=3,
        )
        self.assertFalse(result["complete"])
        comparison = result["kernel_comparisons"][0]
        self.assertEqual(comparison["missing_ranks"], [1])
        self.assertTrue(comparison["count_mismatch"])


if __name__ == "__main__":
    unittest.main()
