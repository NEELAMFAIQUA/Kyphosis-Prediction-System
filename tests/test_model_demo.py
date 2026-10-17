import unittest

import pandas as pd

from model_demo import predict_case, train_kyphosis_model


def synthetic_teaching_rows():
    return pd.DataFrame(
        [
            {
                "Age": 20 + index * 3,
                "Number": 2 + index % 5,
                "Start": 1 + index % 7,
                "Kyphosis": "present" if index % 4 == 0 else "absent",
            }
            for index in range(40)
        ]
    )


class ModelDemoTests(unittest.TestCase):
    def test_random_forest_training_and_prediction(self):
        result = train_kyphosis_model(synthetic_teaching_rows())
        self.assertIn("Age", result["feature_columns"])
        self.assertGreaterEqual(result["accuracy"], 0)
        predicted, probabilities = predict_case(
            result["model"],
            result["feature_columns"],
            {"Age": 45, "Number": 3, "Start": 2},
        )
        self.assertIn(predicted, result["labels"])
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_missing_target_is_rejected(self):
        with self.assertRaises(ValueError):
            train_kyphosis_model(pd.DataFrame({"Age": [1, 2]}))

    def test_single_class_is_rejected(self):
        data = synthetic_teaching_rows().assign(Kyphosis="absent")
        with self.assertRaises(ValueError):
            train_kyphosis_model(data)

    def test_requires_values_for_all_features(self):
        with self.assertRaises(ValueError):
            predict_case(object(), ["Age", "Start"], {"Age": 4})


if __name__ == "__main__":
    unittest.main()
