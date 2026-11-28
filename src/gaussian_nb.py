import numpy as np

class GaussianNB:

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes, n_features = len(self.classes_), X.shape[1]

        var_smoothing = 1e-9

        # calculate the variance of each feature
        feature_variances = np.var(X, axis=0)

        # get the maximum variance
        largest_variance = np.max(feature_variances)

        # calculate the epsilon
        epsilon = var_smoothing * largest_variance

        # Empty arrayes for the means, the variances and the priors
        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes) # one prior per class

        # loop and calculate the means, the variances and the prior for each class
        for idx, k in enumerate(self.classes_):
            Xk = X[y == k] # matrix contains all input data in a specific class k

            self.means_[idx] = Xk.mean(axis=0)
            self.variances_[idx] = Xk.var(axis=0) + epsilon # calculate the variance then add the epsilon
            self.priors_[idx] = Xk.shape[0] / X.shape[0]

        return self

    # calculates the log part
    def _log_gaussian(self, X):
        num = -0.5 * (X[:, None, :] - self.means_)**2 / self.variances_
        log_prob = num - 0.5 * np.log(2 * np.pi * self.variances_)
        return log_prob.sum(axis=2)

    def predict(self, X):
        X = np.asarray(X)

        log_likelihood = self._log_gaussian(X)
        log_prior = np.log(self.priors_)

        return self.classes_[np.argmax(log_likelihood + log_prior, axis=1)]

    
    def predict_proba(self, X):
        X = np.asarray(X)

        # Calculate log likelihood and log prior
        log_likelihood = self._log_gaussian(X)
        log_prior = np.log(self.priors_)

        # Combine them to get the log scores
        scores = log_likelihood + log_prior

        # Subtract the maximum score for numerical stability
        scores = scores - np.max(scores, axis=1, keepdims=True)

        # 4. Convert log scores back
        probabilities = np.exp(scores)

        # 5. Normalize so probabilities sum to 1 for each sample
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)

        return probabilities
