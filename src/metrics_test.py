import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer

from data_split import train_test_split
from gaussian_nb import GaussianNB
from metrics import accuracy_score, confusion_matrix, precision_recall_f1
from sanity_test import load_sanity_dataset


def dataset_metrics(X, y, ax):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    classes = np.unique(y)

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, macro_p, macro_r, macro_f1 = precision_recall_f1(
        y_test, y_pred, classes
    )

    print(f"Accuracy: {acc}")
    print(f"Per-class Prec: {prec}")
    print(f"Per-class Rec: {rec}")
    print(f"Per-class F1: {f1}")
    print(f"Macro (P, R, F1): ({macro_p}, {macro_r}, {macro_f1})\n")

    plot_confusion_matrix(confusion_matrix(y_test, y_pred, classes), classes, ax)


def plot_confusion_matrix(mat, classes, ax):
    im = ax.imshow(mat)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))

    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, round(mat[i, j], 1), ha="center", va="center", color="w")
    return im


def main():
    bunch = load_breast_cancer()
    s_X, s_y = load_sanity_dataset()
    fig, axes = plt.subplots(1, 2)

    dataset_metrics(bunch.data, bunch.target, axes[0])
    dataset_metrics(s_X, s_y, axes[1])
    plt.show()


if __name__ == "__main__":
    main()
