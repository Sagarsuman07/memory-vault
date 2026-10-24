import unittest

from scripts.benchmark_embedding_batches import _parse_batch_sizes, _parity


class EmbeddingBenchmarkHelperTests(unittest.TestCase):
    def test_parse_batch_sizes_accepts_candidates(self):
        self.assertEqual(_parse_batch_sizes("16,32,64"), [16, 32, 64])

    def test_parse_batch_sizes_rejects_nonpositive_values(self):
        with self.assertRaises(Exception):
            _parse_batch_sizes("16,0")

    def test_parse_batch_sizes_rejects_duplicates(self):
        with self.assertRaises(Exception):
            _parse_batch_sizes("16,32,16")

    def test_parity_reports_identical_vectors(self):
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        result = _parity(vectors, vectors)
        self.assertTrue(result["same_dimensions"])
        self.assertEqual(result["max_abs_diff"], 0.0)
        self.assertAlmostEqual(result["min_cosine_similarity"], 1.0)

    def test_parity_detects_dimension_mismatch(self):
        result = _parity([[1.0, 0.0]], [[1.0]])
        self.assertFalse(result["same_dimensions"])
        self.assertIsNone(result["max_abs_diff"])


if __name__ == "__main__":
    unittest.main()
