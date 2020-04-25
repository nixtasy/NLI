import numpy as np



class Evaluation:
    """
    Methods for computing useful regression metrics

    sse: Sum of squared errors
    sst: Total sum of squared errors (actual vs avg(actual))
    r_squared: Regression coefficient (R^2)
    adj_r_squared: Adjusted R^2
    mse: Mean sum of squared errors
    AIC: Akaike information criterion
    BIC: Bayesian information criterion
    """
    def Precision(self,TP, FP):
        if TP == 0:
            return 0
        else:
            return TP / (TP + FP)

    def Recall(self,TP, FN):
        if TP == 0:
            return 0
        else:
            return TP / (TP + FN)

    def F(self,P, R):
        if P == 0:
            return 0
        else:
            return 2 * P * R / (P + R)

    def Accuracy(self,TP, TN, FP, FN):
        return (TP + TN) / (TP + TN + FP + FN)

    def Eva(self, Hn, predictions, labels):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for p, l in zip(predictions, labels):
            if p == Hn:
                if l == p:
                    TP += 1
                else:
                    FP += 1
            else:
                if l == p:
                    TN += 1
                else:
                    FN += 1
        P = self.Precision(TP, FP)
        R = self.Recall(TP, FN)
        F1 = self.F(P, R)
        Acc = self.Accuracy(TP, TN, FP, FN)
        print("H%d: Precison %f, Recall %f, Accuracy %f, F1 %f" % (Hn, P, R, Acc, F1))





class Classifier(
    Evaluation
):
    def __init__(self, fit_intercept=True):
        self.coef_ = None
        self.intercept_ = None
        self.fit_intercept_ = fit_intercept
        self.is_fitted = False
        self.features_ = None
        self.target_ = None

    def __repr__(self):
        return "I am a Linear Regression model!"

    def ingest_data(self, X, y):
        """
       Ingests the given data

        Arguments:
        X: 1D or 2D numpy array
        y: 1D numpy array
        """
        # check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # features and data
        self.features_ = X
        self.target_ = y

    def fit(self, X=None, y=None, fit_intercept_=True):
        """
        Fit model coefficients.
        Arguments:
        X: 1D or 2D numpy array
        y: 1D numpy array
        """

        if X != None:
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            self.features_ = X
        if y != None:
            self.target_ = y

        # degrees of freedom of population dependent variable variance
        self.dft_ = self.features_.shape[0] - 1
        # degrees of freedom of population error variance
        self.dfe_ = self.features_.shape[0] - self.features_.shape[1] - 1

        # add bias if fit_intercept is True
        if self.fit_intercept_:
            X_biased = np.c_[np.ones(self.features_.shape[0]), self.features_]
        else:
            X_biased = self.features_
        # Assign target_ to a local variable y
        y = self.target_

        # closed form solution
        xTx = np.dot(X_biased.T, X_biased)
        inverse_xTx = np.linalg.inv(xTx)
        xTy = np.dot(X_biased.T, y)
        coef = np.dot(inverse_xTx, xTy)

        # set attributes
        if self.fit_intercept_:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0
            self.coef_ = coef

        # Predicted/fitted y
        self.fitted_ = np.dot(self.features_, self.coef_) + self.intercept_

        # Residuals
        residuals = self.target_ - self.fitted_
        self.resid_ = residuals

        # Set is_fitted to True
        self.is_fitted = True

    def fit(self, X=None, y=None, fit_intercept_=True):
        """
        Fits model coefficients.

        Arguments:
        X: 1D or 2D numpy array
        y: 1D numpy array
        fit_intercept: Boolean, whether an intercept term will be included in the fit
        """

        if X != None:
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            self.features_ = X
        if y != None:
            self.target_ = y

        # degrees of freedom of population dependent variable variance
        self.dft_ = self.features_.shape[0] - 1
        # degrees of freedom of population error variance
        self.dfe_ = self.features_.shape[0] - self.features_.shape[1] - 1

        # add bias if fit_intercept is True
        if self.fit_intercept_:
            X_biased = np.c_[np.ones(self.features_.shape[0]), self.features_]
        else:
            X_biased = self.features_
        # Assign target_ to a local variable y
        y = self.target_

        # closed form solution
        xTx = np.dot(X_biased.T, X_biased)
        inverse_xTx = np.linalg.inv(xTx)
        xTy = np.dot(X_biased.T, y)
        coef = np.dot(inverse_xTx, xTy)

        # set attributes
        if self.fit_intercept_:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0
            self.coef_ = coef

        # Predicted/fitted y
        self.fitted_ = np.dot(self.features_, self.coef_) + self.intercept_

        # Residuals
        residuals = self.target_ - self.fitted_
        self.resid_ = residuals

        # Set is_fitted to True
        self.is_fitted = True


    def predict(self, X):
        """Output model prediction.
        Arguments:
        X: 1D or 2D numpy array
        """
        # check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        self.predicted_ = self.intercept_ + np.dot(X, self.coef_)
        return self.predicted_
