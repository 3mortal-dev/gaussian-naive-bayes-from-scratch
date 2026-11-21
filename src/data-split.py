import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# The logic behind `stratified` spliting
# each class must appear in the train and test sets in the same proportion (freq/len()) as in the full data
# don't shuffle data first, instead divide each class into train and test then merge the 2 classes at the end
# this way we garantee that each class appears in the same proportion as it does in the full dataset

def train_test_split(X, y, test_size=0.2, seed=42):

    X, y = np.array(X), np.array(y)
    rand_seed = np.random.RandomState(seed)

    train_list, test_list = [], []
    # divide each class
    for class_ in [0, 1]:
        # brings the current class's index only
        class_part_idx = np.where(y == class_)[0]

        # shuffles each class
        rand_seed.shuffle(class_part_idx)

        N_test = int(test_size * len(class_part_idx))
        test_list.extend(class_part_idx[N_test:])
        train_list.extend(class_part_idx[:N_test])

    train_list = np.array(train_list)
    test_list = np.array(test_list)

    # Now we have 2 lists (train and test) but now they have class 0 at first then class 1, so we have to shuffle each one again
    rand_seed.shuffle(train_list)
    rand_seed.shuffle(test_list)

    return X[train_list], X[test_list], y[train_list], y[test_list]

# returns a `Bunch` object not pd.DataFrame
bunch = load_breast_cancer(as_frame=True)

# now pandas data frame as normal
df = bunch.frame
print(df.head())


X_train, X_test, y_train, y_test = train_test_split(bunch.data, bunch.target)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print("Full :", np.bincount(bunch.target) / len(bunch.target))
print("Train:", np.bincount(y_train) / len(y_train))
print("Test :", np.bincount(y_test) / len(y_test))
