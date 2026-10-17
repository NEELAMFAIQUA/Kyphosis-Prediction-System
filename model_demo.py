"""Synthetic-only classification exercise; it contains no clinical dataset."""

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def evaluate_demo(test_size=0.25, seed=7):
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5")
    features, labels = make_classification(
        n_samples=240,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        class_sep=0.8,
        random_state=seed,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=seed))
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "confusion_matrix": confusion_matrix(y_test, predicted).tolist(),
        "train_size": len(y_train),
        "test_size": len(y_test),
    }
