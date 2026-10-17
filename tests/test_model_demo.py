import unittest

from model_demo import evaluate_demo


class ModelDemoTests(unittest.TestCase):
    def test_evaluation_is_reproducible(self):
        self.assertEqual(evaluate_demo(), evaluate_demo())

    def test_split_sizes_add_to_total(self):
        result = evaluate_demo(0.2)
        self.assertEqual(result["train_size"] + result["test_size"], 240)

    def test_accuracy_is_a_fraction(self):
        self.assertGreaterEqual(evaluate_demo()["accuracy"], 0)
        self.assertLessEqual(evaluate_demo()["accuracy"], 1)

    def test_invalid_split_raises(self):
        with self.assertRaises(ValueError):
            evaluate_demo(0.05)


if __name__ == "__main__":
    unittest.main()
