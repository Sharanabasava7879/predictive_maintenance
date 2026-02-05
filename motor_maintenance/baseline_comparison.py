"""
Baseline Models Comparison with Statistical Validation
Save this file as: baseline_comparison.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix, 
                              roc_auc_score, f1_score, precision_score, 
                              recall_score, make_scorer, accuracy_score)
from scipy import stats
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

class BaselineComparison:
    """
    Compare multiple baseline models with statistical validation
    """
    def __init__(self):
        self.models = {}
        self.results = {}
        self.cv_results = {}
        
    def initialize_models(self):
        """Initialize baseline models"""
        print("\n" + "="*70)
        print("INITIALIZING BASELINE MODELS")
        print("="*70)
        
        # Baseline 1: Logistic Regression
        self.models['Logistic Regression'] = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            solver='lbfgs'
        )
        
        # Baseline 2: Random Forest
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Baseline 3: Try XGBoost, fallback to Gradient Boosting
        try:
            from xgboost import XGBClassifier
            self.models['XGBoost'] = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=10,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
            print("✓ XGBoost available")
        except ImportError:
            self.models['Gradient Boosting'] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
            print("⚠ XGBoost not available, using Gradient Boosting")
        
        print(f"\n✓ Initialized {len(self.models)} baseline models")
        for name in self.models.keys():
            print(f"  • {name}")
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test, 
                          your_model=None, model_name="Your Ensemble"):
        """Train all models and compare performance"""
        print("\n" + "="*70)
        print("TRAINING AND EVALUATING ALL MODELS")
        print("="*70)
        
        all_results = {}
        
        # Train baseline models
        for name, model in self.models.items():
            print(f"\n{'='*70}")
            print(f"Training: {name}")
            print(f"{'='*70}")
            
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            results = self._calculate_metrics(y_test, y_pred, y_pred_proba, train_time)
            all_results[name] = results
            
            print(f"\n✓ {name} trained in {train_time:.2f}s")
            self._print_results(results)
        
        # Evaluate your model if provided
        if your_model is not None:
            print(f"\n{'='*70}")
            print(f"Evaluating: {model_name}")
            print(f"{'='*70}")
            
            start_time = time.time()
            y_pred_proba = your_model.predict_proba(X_test)[:, 1]
            
            # Use optimal threshold if available
            threshold = 0.5
            if hasattr(your_model, 'optimal_threshold'):
                threshold = your_model.optimal_threshold
            elif hasattr(your_model, 'estimators_'):
                # For VotingClassifier, check wrapped models
                for est_name, est in your_model.estimators_:
                    if hasattr(est, 'optimal_threshold'):
                        threshold = est.optimal_threshold
                        break
            
            y_pred = (y_pred_proba >= threshold).astype(int)
            eval_time = time.time() - start_time
            
            results = self._calculate_metrics(y_test, y_pred, y_pred_proba, eval_time)
            all_results[model_name] = results
            
            print(f"\n✓ {model_name} evaluated in {eval_time:.2f}s")
            print(f"  Using threshold: {threshold:.4f}")
            self._print_results(results)
        
        self.results = all_results
        return all_results
    
    def cross_validate_all(self, X, y, cv_folds=10, your_model=None, model_name="Your Ensemble"):
        """Perform k-fold cross-validation"""
        print("\n" + "="*70)
        print(f"CROSS-VALIDATION ({cv_folds}-FOLD)")
        print("="*70)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_results = {}
        
        scoring = {
            'f1': make_scorer(f1_score),
            'recall': make_scorer(recall_score),
            'precision': make_scorer(precision_score),
            'roc_auc': make_scorer(roc_auc_score)
        }
        
        # Cross-validate baseline models
        for name, model in self.models.items():
            print(f"\n📊 Cross-validating: {name}...")
            
            cv_scores = {}
            for metric_name, scorer in scoring.items():
                scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
                cv_scores[metric_name] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores
                }
            
            cv_results[name] = cv_scores
            
            print(f"  F1:        {cv_scores['f1']['mean']:.4f} ± {cv_scores['f1']['std']:.4f}")
            print(f"  Recall:    {cv_scores['recall']['mean']:.4f} ± {cv_scores['recall']['std']:.4f}")
            print(f"  Precision: {cv_scores['precision']['mean']:.4f} ± {cv_scores['precision']['std']:.4f}")
            print(f"  ROC-AUC:   {cv_scores['roc_auc']['mean']:.4f} ± {cv_scores['roc_auc']['std']:.4f}")
        
        # Cross-validate your model
        if your_model is not None:
            print(f"\n📊 Cross-validating: {model_name}...")
            
            cv_scores = {}
            for metric_name, scorer in scoring.items():
                scores = cross_val_score(your_model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
                cv_scores[metric_name] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'scores': scores
                }
            
            cv_results[model_name] = cv_scores
            
            print(f"  F1:        {cv_scores['f1']['mean']:.4f} ± {cv_scores['f1']['std']:.4f}")
            print(f"  Recall:    {cv_scores['recall']['mean']:.4f} ± {cv_scores['recall']['std']:.4f}")
            print(f"  Precision: {cv_scores['precision']['mean']:.4f} ± {cv_scores['precision']['std']:.4f}")
            print(f"  ROC-AUC:   {cv_scores['roc_auc']['mean']:.4f} ± {cv_scores['roc_auc']['std']:.4f}")
        
        self.cv_results = cv_results
        return cv_results
    
    def statistical_tests(self, metric='f1', your_model_name="Your Ensemble"):
        """Perform statistical significance tests"""
        print("\n" + "="*70)
        print(f"STATISTICAL TESTS ({metric.upper()})")
        print("="*70)
        
        if not self.cv_results:
            print("⚠ Run cross_validate_all() first!")
            return None
        
        if your_model_name not in self.cv_results:
            print(f"⚠ {your_model_name} not found!")
            return None
        
        your_scores = self.cv_results[your_model_name][metric]['scores']
        results = {}
        
        for name, cv_data in self.cv_results.items():
            if name == your_model_name:
                continue
            
            baseline_scores = cv_data[metric]['scores']
            
            # Paired t-test
            t_statistic, p_value = stats.ttest_rel(your_scores, baseline_scores)
            
            mean_diff = your_scores.mean() - baseline_scores.mean()
            pooled_std = np.sqrt((your_scores.std()**2 + baseline_scores.std()**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            is_significant = p_value < 0.05
            is_better = mean_diff > 0
            
            results[name] = {
                't_statistic': t_statistic,
                'p_value': p_value,
                'mean_difference': mean_diff,
                'cohens_d': cohens_d,
                'is_significant': is_significant,
                'is_better': is_better
            }
            
            print(f"\n  vs {name}:")
            print(f"    Difference: {mean_diff:+.4f}")
            print(f"    p-value: {p_value:.4f}")
            print(f"    Cohen's d: {cohens_d:.4f}")
            
            if is_significant and is_better:
                print(f"    ✅ Significantly BETTER (p < 0.05)")
            elif is_significant:
                print(f"    ❌ Significantly worse")
            else:
                print(f"    ⚠️ No significant difference")
        
        return results
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba, training_time):
        """Calculate metrics"""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'f1': f1_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred) if (tp + fp) > 0 else 0,
            'accuracy': accuracy_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'confusion_matrix': cm,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'training_time': training_time
        }
    
    def _print_results(self, results):
        """Print results"""
        print(f"\n  Metrics:")
        print(f"    F1:        {results['f1']:.4f}")
        print(f"    Recall:    {results['recall']:.4f}")
        print(f"    Precision: {results['precision']:.4f}")
        print(f"    Accuracy:  {results['accuracy']:.4f}")
        print(f"    ROC-AUC:   {results['roc_auc']:.4f}")
        print(f"    FN: {results['fn']}, FP: {results['fp']}")