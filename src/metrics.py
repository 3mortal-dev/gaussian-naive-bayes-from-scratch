import numpy as np
from sklearn.metrics import recall_score

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

# TPR(Recall) = TP / (TP+FN)
# FPR = FP / (FP+TN)
# percision = TP / (TP + FP)
# f1 = (2 * precision * recall) / (precision + recall)
# we'll get all of this shit from the confusion matrix

def precision_recall_f1(y, y_pred, classes):
    confusion_mat = confusion_matrix(y, y_pred,  classes)

    # For each class true means (Either I got a true prediction or I truely excluded that class) (predicting 2 while the true class is 3, from class 0 prespective is true)
    # For each class false means (My prediction was wrong)
    # For each class positive means (I predicted that class)
    # For each class negative means (I didn't predict that class)

    TP = np.diag(confusion_mat)
    FP = np.sum(confusion_mat, axis=0) - TP # FP for each class is the sum of its column except the value when it's true (diag)
    FN = np.sum(confusion_mat, axis=1) - TP # FN for each class is the sum of its row except the value when it's true (diag)

    # np.divide() handles 0 division case and it also divides the 2 matrices effeciently
    precision = np.divide(
        TP, TP + FP, out=np.zeros_like(TP, dtype=float), where=(TP + FP) > 0
    )
    recall = np.divide(
        TP, TP + FN, out=np.zeros_like(TP, dtype=float), where=(TP + FN) > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(TP, dtype=float),
        where=(precision + recall) > 0,
      )

    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    return precision, recall, f1, macro_precision, macro_recall, macro_f1

def testing():
    y = [0, 1, 0, 1]
    y_pred = [0, 0, 0, 1]
    print(accuracy_score(y, y_pred))
    print(confusion_matrix(y, y_pred, [0,1]))

if __name__ == '__main__':
    testing()


