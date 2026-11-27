import numpy as np

class GaussianNB:

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes, n_features = len(self.classes_), X.shape[1]

        # Empty arrayes for the means, the variances and the priors
        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes) # one prior per class

        # loop and calculate the means, the variances and the prior for each class
        for idx, k in enumerate(self.classes_):
            Xk = X[y == k] # matrix contains all input data in a specific class k

            self.means_[idx] = Xk.mean(axis=0)
            self.variances_[idx] = np.maximum(Xk.var(axis=0), 1e-9)
            self.priors_[idx] = Xk.shape[0] / X.shape[0]

        return self

    # calculates the log part
    def _log_gaussian(self, X):
        num = -0.5 * (X[:, None, :] - self.means_)**2 / self.variances_
        log_prob = num - 0.5 * np.log(2 * np.pi * self.variances_)
        return log_prob.sum(axis=2)

    def predict_proba(self, X):
        # P(x, y)
        joint_log_prob = self._log_gaussian(X) + np.log(self.priors_)


        # P(x, y) for the winning class
        max_log = np.max(joint_log_prob, axis=1, keepdims=True)
        exp_shifted = np.exp(joint_log_prob - max_log)
        return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

    def predict(self, X):
        X = np.asarray(X)

        log_likelihood = self._log_gaussian(X)
        log_prior = np.log(self.priors_)

        return self.classes_[np.argmax(log_likelihood + log_prior, axis=1)]
