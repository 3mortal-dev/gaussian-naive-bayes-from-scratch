from sklearn.datasets import load_breast_cancer
from metrics import accuracy_score, confusion_matrix

import numpy as np
from data_split import train_test_split
from gaussian_nb import GaussianNB


def main():

    # Load dataset
    dataset = load_breast_cancer(as_frame=True)

    X, y = dataset.data, dataset.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, seed=42
    )

    # Train model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Training samples: {len(y_train)}")
    print(f"Testing samples: {len(y_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(confusion_matrix(y_test, y_pred, np.unique(y)))


if __name__ == "__main__":
    main()
