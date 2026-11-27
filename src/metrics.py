import numpy as np

# classifier's accuracy is the ratio between its correct classifications / total classifications

def accuracy_score(y, y_pred):
    # return np.sum([y == y_pred]) / len(y)

    y = np.asarray(y)
    y_pred = np.asarray(y_pred)
    assert len(y) == len(y_pred)

    filter = [y == y_pred]
    correct_classifications = np.sum(filter)
    total_classifications = len(y)

    return correct_classifications / total_classifications

# confusion matrix of a classifier is an nxn matrix (2x2 for a b-classifier) which's represented as:
# y\y_hat     0     1
# 0         (TN), (FP)
# 1         (FN), (TP)

def confusion_matrix(y, y_pred, classes):

    y = np.asarray(y)
    y_pred = np.asarray(y_pred)
    classes = np.asarray(classes)
    assert len(y) == len(y_pred)

    confusion_mat = np.zeros((len(classes), len(classes)), dtype=float)

    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            confusion_mat[i][j] = np.sum([(y_pred == pred_class) & (y == true_class)])

    # confusion_mat /= len(y)

    return confusion_mat
# now TP for each class is located at confusion matrix's diagonal

def testing():
    y = [0, 1, 0, 1]
    y_pred = [0, 0, 0, 1]
    print(accuracy_score(y, y_pred))
    print(confusion_matrix(y, y_pred, [0,1]))

if __name__ == '__main__':
    testing()


