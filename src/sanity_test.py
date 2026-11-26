import numpy as np

from data_split import train_test_split
from gaussian_nb import GaussianNB


def test_gnb_sanity_check():
    X = np.array([85, 92, 74, 111, 82, 73, 88, 75, 88, 75, 78]).reshape(-1, 1)
    y = np.array([1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0])

    # Split dataset
    X_train, _, y_train, _ = train_test_split(X, y)

    # Train model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # print(X_train.reshape(-1)[y_train==0], X_test.reshape(-1)[y_test==0])
    # print(X_train.reshape(-1)[y_train==1], X_test.reshape(-1)[y_test==1])

    # ASSUME THAT I CALCULATED THEM WITH PEN AND PAPER :)
    expected_means = np.array([
        X_train[y_train == c].mean(axis=0)
        for c in model.classes_
    ])
    expected_variances = np.array([
        X_train[y_train == c].var(axis=0)
        for c in model.classes_
    ])
    expected_priors = np.array([
        np.mean(y_train == c)
        for c in model.classes_
    ])

    np.testing.assert_allclose(model.means_, expected_means)
    np.testing.assert_allclose(model.variances_, expected_variances)
    np.testing.assert_allclose(model.priors_, expected_priors)
    print("All good!")
    # print(f"\n {np.log(prior_0) + model._log_gaussian(X)}")



if __name__ == "__main__":
    test_gnb_sanity_check()
