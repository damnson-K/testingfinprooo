class AdjustedThresholdModel:
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold
    
    def predict(self, X):
        """Predict using adjusted threshold"""
        probabilities = self.model.predict_proba(X)[:, 1]  # Get probabilities for class 1
        return (probabilities >= self.threshold).astype(int)  # Apply threshold

    def predict_proba(self, X):
        """Return original probability scores"""
        return self.model.predict_proba(X)