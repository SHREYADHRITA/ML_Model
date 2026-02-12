from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


try:
    from xgboost import XGBClassifier
    _has_xgb = True
except Exception:
    _has_xgb = False

class BaseModel:
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        return self._clf.predict(X)

    def predict_proba(self, X):
        if hasattr(self._clf, "predict_proba"):
            return self._clf.predict_proba(X)
        raise AttributeError("Underlying classifier does not support predict_proba")

    @property
    def name(self):
        return self.__class__.__name__

class LogisticRegressionModel(BaseModel):
    def __init__(self, C=1.0, max_iter=200, random_state=42):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self._clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=self.random_state))
        ])

    def fit(self, X, y):
        self._clf.fit(X, y)
        return self

class DecisionTreeModel(BaseModel):
    def __init__(self, max_depth=None, random_state=42):
        self.max_depth = max_depth
        self.random_state = random_state
        self._clf = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)

    def fit(self, X, y):
        self._clf.fit(X, y)
        return self

class KNNModel(BaseModel):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self._clf = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=self.n_neighbors))])

    def fit(self, X, y):
        self._clf.fit(X, y)
        return self

class NaiveBayesModel(BaseModel):
    def __init__(self):
        self._clf = GaussianNB()

    def fit(self, X, y):
        self._clf.fit(X, y)
        return self

class RandomForestModel(BaseModel):
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._clf = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.random_state)

    def fit(self, X, y):
        self._clf.fit(X, y)
        return self

class XGBoostModel(BaseModel):
    def __init__(self, n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False):
        if not _has_xgb:
            raise ImportError("xgboost is not installed. Install with `pip install xgboost` to use this class.")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        # disable label encoder warning in newer xgboost
        self._clf = XGBClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate, random_state=self.random_state, use_label_encoder=False, eval_metric='logloss')

    def fit(self, X, y):
        self._clf.fit(X, y)
        return self

