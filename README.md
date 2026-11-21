# Gaussian Naive Bayes from Scratch

**Programming Assignment**  
**Machine Learning Course** | Individual or pairs | **Language:** Python 3 (NumPy)

---

## 1. Objective

In this assignment, you will implement a **Gaussian Naive Bayes (GNB)** classifier from scratch using only NumPy, evaluate it with metrics you write yourself, run a few experiments to understand how it behaves, and finally verify your implementation against `scikit-learn`.

After finishing, you should be able to:
* Explain Bayes' theorem and the "naive" conditional independence assumption.
* Estimate class priors and per-class Gaussian parameters from data.
* Compute predictions in log-space and explain why this is necessary.
* Evaluate a classifier with a confusion matrix, precision, recall, and $F_1$ score.
* Recognize when the naive assumption helps and when it hurts.

---

## 2. Background

For a sample $x = (x_1, \dots, x_d)$ and a class $c$, Bayes' theorem gives the posterior probability:

$$P(c \mid x) = \frac{P(x \mid c) P(c)}{P(x)}$$

The **naive assumption** states that the features are conditionally independent given the class, so the likelihood factorizes:

$$P(x \mid c) = P(x_1 \mid c) \cdot P(x_2 \mid c) \cdots P(x_d \mid c) = \prod_{j=1}^{d} P(x_j \mid c)$$

In the Gaussian version, each feature $j$ of class $c$ follows a normal distribution with its own mean ($\mu_{cj}$) and variance ($\sigma^2_{cj}$), both estimated from the training samples of that class:

$$P(x_j \mid c) = \frac{1}{\sqrt{2\pi\sigma^2_{cj}}} \exp\left(-\frac{(x_j - \mu_{cj})^2}{2\sigma^2_{cj}}\right)$$

$$\text{prior}(c) = \frac{\text{number of samples in class } c}{\text{total samples}}$$

Multiplying many small probabilities underflows to zero, so we work with logarithms. Since $P(x)$ is the same for every class, the predicted class $\hat{y}$ is:

$$\hat{y} = \arg\max_c \left[ \log \text{prior}(c) + \sum_{j=1}^{d} \log P(x_j \mid c) \right]$$

where:

$$\log P(x_j \mid c) = -\frac{1}{2} \log(2\pi\sigma^2_{cj}) - \frac{(x_j - \mu_{cj})^2}{2\sigma^2_{cj}}$$

**Variance smoothing:** If a feature is constant inside a class, its variance is $0$ and the formula divides by zero. Add a small value $\epsilon = \text{var\_smoothing} \times (\text{largest variance of any feature in the whole training set } X)$ to all variances. The default `var_smoothing` is $10^{-9}$ (`1e-9`).

---

## 3. Datasets and Allowed Libraries

| Dataset | Details |
| :--- | :--- |
| **Breast Cancer Wisconsin** | $569$ samples, $30$ features, $2$ classes ($212$ malignant, $357$ benign).<br>Load with `sklearn.datasets.load_breast_cancer()`. |

* **Allowed:** `numpy`, `matplotlib`, `pandas` (optional), and `sklearn.datasets` *only* to load the data.
* **Allowed only in Task 6:** `sklearn.naive_bayes.GaussianNB`, to compare with your results.
* **Not allowed:** `sklearn.metrics`, `train_test_split`, or any ready-made classifier inside your own code. Write them yourself.
* **Reproducibility:** Use `np.random.seed(42)` everywhere.

---

## 4. Tasks

### Task 1: Data Preparation `[10 marks]`
* Write a function `train_test_split(X, y, test_size=0.2, seed=42)` that shuffles the data and returns `X_train, X_test, y_train, y_test`.
* Make it **stratified**: each class must appear in the train and test sets in the same proportion as in the full data. Print the class counts of both sets to prove it.
* **Question:** Does GNB need feature scaling? Explain in one or two sentences after testing it (compare accuracy with and without standardization).

### Task 2: Implement Gaussian Naive Bayes `[35 marks]`
* Implement the Gaussian Naive Bayes classifier from scratch using NumPy.
* **Sanity test (required in your notebook):** Create a tiny dataset by hand with $2$ classes and $1$ feature, compute the mean, variance, and one posterior with a calculator, and show that your class gives the same numbers.

### Task 3: Evaluation Metrics from Scratch `[10 marks]`
Implement the following evaluation functions:
* `accuracy(y_true, y_pred)`
* `confusion_matrix(y_true, y_pred, classes)` returning a $k \times k$ array (`rows` = true class, `columns` = predicted class).
* `precision_recall_f1(y_true, y_pred, classes)` returning the per-class values and the macro average. Handle the case of division by zero (a class that is never predicted) by returning `0`.

Train on the dataset(s), report all metrics on the test set, and display each confusion matrix as a `matplotlib` heatmap.

### Task 4: Experiments `[20 marks]`
* **(a) Breaking the naive assumption:** Duplicate one strongly informative feature of the Breast Cancer dataset several times ($1, 3, 5, 10$ copies) and retrain. Plot accuracy and the average maximum predicted probability. Explain what happens to the model's confidence.

### Task 5: Visualization `[10 marks]`
* Plot the fitted Gaussian curves of every class (one figure per feature, with the training points as dots on the x-axis).
* Train a GNB model using 2 features and plot its decision regions on a mesh grid with the test points on top.
* Describe in two sentences why the decision boundaries are curved and not straight lines.

### Task 6: Compare with scikit-learn `[5 marks]`
* Train `sklearn.naive_bayes.GaussianNB` with the same `var_smoothing` on the dataset(s).
* Show with `np.allclose` that your `theta_`, `var_`, and `class_prior_` match `sklearn`'s, and that `predict_proba` matches.
* Report the percentage of test samples where the two models agree (it should be $100\%$). If something differs, find out why.

### Task 7: Short Questions `[10 marks]`
Answer each question in **3 to 5 sentences** in your report:
1. Why is the classifier called "naive"? Give an example where the assumption is clearly false.
2. Why can we ignore $P(x)$ when predicting the class?
3. What happens if a feature has zero variance within a class and there is no smoothing?
4. How is GNB related to Quadratic Discriminant Analysis (QDA)? What extra assumption does GNB make?
5. GNB trains in one pass over the data. What is its time complexity for training and for predicting one sample (in terms of $n$, $d$, and $k$)?

---

## 5. Deliverables & Policies

* **Notebook:** One `.ipynb` file that runs from top to bottom without errors, with a code cell and a short explanation for every task.
* **Report:** A PDF of at most 4 pages with your figures, tables of results, and answers to Task 7 (you may export it from the notebook).
* **File Names:** `GNB_ID1_ID2.ipynb` and `GNB_ID1_ID2.pdf`, zipped together as `GNB_ID1_ID2.zip`.
* **Header:** The first cell of the notebook must contain the full names and IDs of all members.
* **Integrity:** Starter code in this document may be used freely. The rest must be your own work. All submissions go through automated plagiarism detection. You may discuss ideas, but never share code.
* **Penalties:** Marks are deducted for loops over samples in `predict`, use of forbidden libraries, code that does not run, and plots without titles, axis labels, or legends.
