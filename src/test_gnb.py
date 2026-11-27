
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB

from data_split import train_test_split
from gaussian_nb import GaussianNB
from metrics import accuracy_score, confusion_matrix


def main():

    # Load dataset
    dataset = load_breast_cancer(as_frame=True)
    X, y = dataset.data, dataset.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, seed=42
    )

    # Train custom model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Train sklearn model
    sklearn_model = SklearnGaussianNB(var_smoothing=1e-9)
    sklearn_model.fit(X_train, y_train)

    # Evaluate custom model
    y_pred = model.predict(X_test)

    print("--- Custom Model ---")
    print(f"Training samples: {len(y_train)}")
    print(f"Testing samples: {len(y_test)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, np.unique(y)))

    # Compare learned parameters
    print("\n--- Parameter Comparison ---")
    print("Means:", np.allclose(model.means_, sklearn_model.theta_))
    print("Variances:", np.allclose(model.variances_, sklearn_model.var_))
    print("Priors:", np.allclose(model.priors_, sklearn_model.class_prior_))

    # Compare predicted probabilities
    custom_proba = model.predict_proba(X_test)
    sklearn_proba = sklearn_model.predict_proba(X_test)

    print("\nProbabilities match:",
          np.allclose(custom_proba, sklearn_proba))

    # Compare predictions
    sklearn_pred = sklearn_model.predict(X_test)
    agreement = np.mean(y_pred == sklearn_pred) * 100

    print(f"Prediction agreement: {agreement:.2f}%")


if __name__ == "__main__":
    main()
