import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


class LegislacaoModelPipeline:
    def __init__(self, base_path="data"):
        self.base_path = base_path
        self.processed_path = os.path.join(base_path, "processed")
        self.models_path = os.path.join(base_path, "models")
        self.features_file = os.path.join(self.processed_path, "leis_features.csv")
        self.metrics_file = os.path.join(self.processed_path, "model_metrics.json")
        
        os.makedirs(self.models_path, exist_ok=True)
        
        self.feature_cols = [
            "flesch",
            "palavras_por_sentenca",
            "sentencas_por_artigo",
            "densidade_referencias",
            "ttr",
            "proporcao_termos_tecnicos"
        ]

    def build_classifiers(self):
        return {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "svm": SVC(probability=True, kernel="rbf", random_state=42),
            "xgboost": XGBClassifier(eval_metric="logloss", random_state=42)
        }

    def train_and_evaluate(self, X, y, target_name):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        classifiers = self.build_classifiers()
        results = {}
        best_model = None
        best_f1 = -1.0
        best_model_name = ""
        
        for name, clf in classifiers.items():
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", clf)
            ])
            
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1_macro")
            
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average="macro")
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred).tolist()
            
            results[name] = {
                "cv_f1_macro_mean": float(np.mean(cv_scores)),
                "test_accuracy": float(acc),
                "test_f1_macro": float(f1_macro),
                "confusion_matrix": cm,
                "precision_class_1": float(report.get("1", {}).get("precision", 0.0) if "1" in report else report.get("1.0", {}).get("precision", 0.0)),
                "recall_class_1": float(report.get("1", {}).get("recall", 0.0) if "1" in report else report.get("1.0", {}).get("recall", 0.0))
            }
            
            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_model = pipeline
                best_model_name = name
                
        if best_model:
            model_filename = f"model_{target_name}.joblib"
            joblib.dump(best_model, os.path.join(self.models_path, model_filename))
            
        return results, best_model_name

    def run_pipeline(self):
        if not os.path.exists(self.features_file):
            return False
            
        df = pd.read_csv(self.features_file).drop_duplicates(subset=["id"])
        
        X = df[self.feature_cols]
        
        metrics = {}
        
        for target in ["teve_veto", "instavel"]:
            y = df[target]
            
            if len(np.unique(y)) < 2:
                continue
                
            results, best_name = self.train_and_evaluate(X, y, target)
            metrics[target] = {
                "best_model": best_name,
                "classifiers": results
            }
            
        if metrics:
            with open(self.metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=4)
            return True
        return False

if __name__ == "__main__":
    pipeline = LegislacaoModelPipeline()
    success = pipeline.run_pipeline()
    if success:
        print("Training and evaluation complete.")
    else:
        print("Modeling pipeline failed or insufficient data.")
