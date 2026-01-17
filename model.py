import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class train_model():
    """ This class implements a function to train a model on the given dataset """
    def __init__(self, dataset_path, target_column):
        self.dataset = pd.read_csv(dataset_path)
        self.target = target_column


    def feature_engineering(self):
        """ This class is specific to the dataset we have chosen.
        This class will implement the basic feature engineering to
        modify the dataset for model calculation """

        encoder = LabelEncoder()
        for col in self.dataset.columns:
            self.dataset[col] = encoder.fit_transform(self.dataset[col])

        X = self.dataset.drop(["class"], axis=1)
        Y = self.dataset["class"]

        return X, Y

    def logistic_regression(self, X, y, test_size=0.2, random_state=42, **kwargs):
        """ This method implements a simple linear regression model """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred)
        }

        return model, metrics

    def decision_tree_classifier(self, X, y, test_size=0.2, random_state=42, **kwargs):
        """ This method implements a decision tree classifier """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = DecisionTreeClassifier(random_state=random_state, **kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred)
        }

        return model, metrics

    def k_nearest_neighbor_classifier(self, X, y, test_size=0.2, random_state=42, **kwargs):
        """ This method implements a k-nearest neighbor classifier """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = KNeighborsClassifier(**kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred)
        }

        return model, metrics

    def naive_bayes_classifier(self, X, y, test_size=0.2, random_state=42, **kwargs):
        """ This method implements a naive bayes classifier """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = GaussianNB(**kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred)
        }

        return model, metrics

    def random_forest(self, X, y, test_size=0.2, random_state=42, **kwargs):
        """ This method implements a random forest model """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = RandomForestClassifier(random_state=random_state, **kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred)
        }

        return model, metrics

    def xgboost_model(self, X, y, test_size=0.2, random_state=42, **kwargs):
        """ This method implements an XGBoost model """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=random_state,
            **kwargs
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred)
        }

        return model, metrics

    def model_metrics(self):
        """ This method computes various model metrics """
