from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class RemoveZeroVarianceFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.non_zero_variance_features_ = np.var(X, axis=0) > 0
        return self
    
    def transform(self, X, y=None):
        return X[:, self.non_zero_variance_features_]

class RemoveAutocorrelatedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.95):
        self.threshold = threshold
    
    def fit(self, X, y=None):
        corr_matrix = np.corrcoef(X, rowvar=False)
        upper_triangle_indices = np.triu_indices_from(corr_matrix, k=1)
        self.to_remove_ = set()
        
        for i, j in zip(*upper_triangle_indices):
            if abs(corr_matrix[i, j]) > self.threshold:
                self.to_remove_.add(j)
        
        return self
    
    def transform(self, X, y=None):
        features_to_keep = [i for i in range(X.shape[1]) if i not in self.to_remove_]
        return X[:, features_to_keep]
