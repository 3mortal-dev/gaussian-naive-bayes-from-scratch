import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from gaussian_nb import GaussianNB
from data_split import train_test_split

from pathlib import Path

Path("reports/figures/features").mkdir(parents=True, exist_ok=True)


def gaussian_pdf(x, mean, var):
    return (1.0 / np.sqrt(2.0 * np.pi * var)) * np.exp(-0.5 * ((x - mean) ** 2) / var)


def plot_fitted_gaussians(model, X_train, y_train):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    n_features = X_train.shape[1]

    print("Saving...")
    for j in range(n_features):
        fig, ax = plt.subplots(figsize=(7, 4))

        col = X_train[:, j]
        x_grid = np.linspace(col.min(), col.max(), 300)

        for idx, class_ in enumerate(model.classes_):
            mean = model.means_[idx, j]
            var = model.variances_[idx, j]
            pdf_vals = gaussian_pdf(x_grid, mean, var)

            # Plot fitted Gaussian curve
            ax.plot(x_grid, pdf_vals, label=f"Class {class_}")
            # Plot training points for this class as dots on the x-axis (y = 0)
            cls_points = X_train[y_train == class_, j]
            ax.scatter(cls_points, np.zeros_like(cls_points), alpha=0.6, zorder=3)

        ax.set_title(f"Fitted Gaussians — {j}")
        fig.savefig(f"reports/figures/features/feat - {j}.png")
    print("Figures saved to reports/figures/features/feat/")


def plot_decision_regions(model, X_test, y_test):
    xx, yy = np.meshgrid(
        np.linspace(X_test[:, 0].min() - 1, X_test[:, 0].max() + 1, 200),
        np.linspace(X_test[:, 1].min() - 1, X_test[:, 1].max() + 1, 200),
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots()

    ax.contourf(xx, yy, Z, alpha=0.2)
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors="k")
    ax.set_title("GaussianNB Decision Regions")

    fig.savefig("reports/desicion_region.png")

def main():
    X, y = load_breast_cancer(return_X_y=True)

    X_train, _, y_train, _ = train_test_split(X, y)
    model = GaussianNB()
    model.fit(X_train, y_train)
    plot_fitted_gaussians(model, X_train, y_train)

    f0, f1 = 0, 1 # up to 29 (30 features)
    X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X[:, [f0, f1]], y)
    model_2d = GaussianNB()
    model_2d.fit(X_train_2d, y_train_2d)
    plot_decision_regions(model_2d, X_test_2d, y_test_2d)


if __name__ == "__main__":
    main()
