import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer

from data_split import train_test_split
from gaussian_nb import GaussianNB
from metrics import accuracy_score, confusion_matrix, precision_recall_f1

def smt(X_train, X_test, y_train, y_test, best_idx, k):
    train_col = X_train[:, [best_idx]]
    test_col = X_test[:, [best_idx]]

    X_train_dup = np.hstack((X_train, np.repeat(train_col, k, axis=1)))
    X_test_dup = np.hstack((X_test, np.repeat(test_col, k, axis=1)))

    model = GaussianNB()
    model.fit(X_train_dup, y_train)

    y_pred = model.predict(X_test_dup)

    acc = accuracy_score(y_test, y_pred)
    probs = model.predict_proba(X_test_dup)


    return acc, np.mean(np.max(probs, axis=1))

def main():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    corrs = [np.abs(np.corrcoef(X_train[:, j], y_train)[0, 1]) for j in range(X_train.shape[1])]
    idx_max_related_feature = np.argmax(corrs)

    accs, max_probs = [], []

    copies = [1, 3, 5, 10]

    for k in copies:
        acc, avg_prob = smt(X_train, X_test, y_train, y_test, idx_max_related_feature, k)
        accs.append(acc)
        max_probs.append(avg_prob)
        print(f"k={k:2d} | Accuracy: {acc:.4f} | Avg Max Prob: {avg_prob:.4f}")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(copies, accs, marker="o")
    ax.plot(copies, max_probs, marker="s")

    ax.set_xticks(copies)
    ax.grid(True)

    fig.savefig("reports/figures/acc_and_avg_max_predicted_prob.png")
    plt.show()
main()
