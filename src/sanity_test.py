import numpy as np

from data_split import train_test_split
from gaussian_nb import GaussianNB


def main():
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
    calc_mean_0 = X_train[y_train==0].mean()
    calc_mean_1 = X_train[y_train==1].mean()
    print(f"Calculated mean for both classes is {calc_mean_0, calc_mean_1}")
    print(f"Mean is: {model.means_.reshape(-1)}\n")

    calc_variance_0 = X_train[y_train==0].var()
    calc_variance_1 = X_train[y_train==1].var()
    print(f"Calculated variance for both classes is {calc_variance_0, calc_variance_1}")
    print(f"Variance is: {model.variances_.reshape(-1)}\n")

    prior_0 = len(y_train[y_train==0]) / len(y_train)
    prior_1 = len(y_train[y_train==1]) / len(y_train)
    print(f"Calculated prior for both classes is {prior_0, prior_1}")
    print(f"Prior is: {model.priors_.reshape(-1)}")
    # print(f"\n {np.log(prior_0) + model._log_gaussian(X)}")



if __name__ == "__main__":
    main()
