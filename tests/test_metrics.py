import unittest

from app.eeg_epilepsy.metrics import confusion_counts, metric_report, normalize_label, threshold_probabilities


class MetricTests(unittest.TestCase):
    def test_normalize_label_accepts_project_terms(self):
        self.assertEqual(normalize_label("ictal"), 1)
        self.assertEqual(normalize_label("seizure"), 1)
        self.assertEqual(normalize_label("non-ictal"), 0)
        self.assertEqual(normalize_label("normal"), 0)

    def test_metric_report_uses_seizure_as_positive_class(self):
        report = metric_report(
            y_true=["seizure", "seizure", "normal", "normal"],
            y_pred=["seizure", "normal", "seizure", "normal"],
        )

        self.assertEqual(report.confusion.true_positive, 1)
        self.assertEqual(report.confusion.false_negative, 1)
        self.assertEqual(report.confusion.false_positive, 1)
        self.assertEqual(report.confusion.true_negative, 1)
        self.assertEqual(report.accuracy, 0.5)
        self.assertEqual(report.precision, 0.5)
        self.assertEqual(report.recall, 0.5)
        self.assertEqual(report.f1_score, 0.5)

    def test_threshold_probabilities_is_inclusive_at_threshold(self):
        self.assertEqual(threshold_probabilities([0.49, 0.5, 0.9], threshold=0.5), [0, 1, 1])

    def test_confusion_counts_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            confusion_counts([1, 0], [1])


if __name__ == "__main__":
    unittest.main()
