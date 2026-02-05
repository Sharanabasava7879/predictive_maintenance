import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                              roc_auc_score, f1_score, precision_score, 
                              recall_score, precision_recall_curve)
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
import joblib
import warnings
warnings.filterwarnings('ignore')

class PredictiveMaintenanceModel:
    def __init__(self):
        self.failure_model = None
        self.optimal_threshold = 0.5
        
    def train_failure_model(self, X_train, y_train, handle_imbalance=True):
        """
        BALANCED OPTIMIZED MODEL
        Target: F1 ≥ 0.65, Recall ≥ 0.98, FN ≤ 2, FP < 200, Precision > 0.35
        """
        print("\n" + "="*70)
        print("TRAINING BALANCED OPTIMIZED MODEL")
        print("="*70)
        
        print(f"\n📊 Original Data:")
        n_failures = sum(y_train)
        n_normal = len(y_train) - n_failures
        print(f"   Failures: {n_failures} ({n_failures/len(y_train)*100:.2f}%)")
        print(f"   Normal: {n_normal} ({n_normal/len(y_train)*100:.2f}%)")
        print(f"   Imbalance: {n_normal/n_failures:.1f}:1")
        
        if handle_imbalance:
            print("\n🔄 Strategic Resampling for Balanced Performance")
            print("   Strategy: Moderate SMOTE + Controlled Undersampling + Tomek Cleaning")
            
            # Stage 1: Moderate SMOTE to 30% of majority
            target_minority = int(n_normal * 0.30)
            smote = SMOTE(sampling_strategy={1: target_minority}, k_neighbors=5, random_state=42)
            X_smote, y_smote = smote.fit_resample(X_train, y_train)
            
            print(f"   After SMOTE: Failures={sum(y_smote)}, Normal={len(y_smote)-sum(y_smote)}")
            
            # Stage 2: Conservative undersampling to 2.5:1 ratio
            target_majority = int(target_minority * 2.5)
            rus = RandomUnderSampler(sampling_strategy={0: target_majority}, random_state=42)
            X_balanced, y_balanced = rus.fit_resample(X_smote, y_smote)
            
            print(f"   After Undersampling: Failures={sum(y_balanced)}, Normal={len(y_balanced)-sum(y_balanced)}")
            
            # Stage 3: Tomek Links for boundary cleaning
            tomek = TomekLinks(sampling_strategy='all')
            X_final, y_final = tomek.fit_resample(X_balanced, y_balanced)
            
            print(f"   After Tomek Cleaning: Failures={sum(y_final)}, Normal={len(y_final)-sum(y_final)}")
            print(f"   Final Ratio: {(len(y_final)-sum(y_final))/sum(y_final):.2f}:1")
        else:
            X_final, y_final = X_train, y_train
        
        print("\n🚀 Building Precision-Recall Balanced Ensemble")
        print("   Architecture: GB (pattern) + RF (robust) + DT (recall boost)")
        
        # Model 1: Gradient Boosting - primary predictor
        gb_model = GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.08,
            max_depth=7,
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.80,
            max_features='sqrt',
            random_state=42,
            verbose=0
        )
        
        # Model 2: Random Forest - balanced approach
        rf_model = RandomForestClassifier(
            n_estimators=250,
            max_depth=14,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features='sqrt',
            class_weight={0: 1, 1: 2.5},
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        # Model 3: Decision Tree - high recall safety net
        dt_model = DecisionTreeClassifier(
            max_depth=15,
            min_samples_split=6,
            min_samples_leaf=2,
            class_weight={0: 1, 1: 4},
            random_state=42
        )
        
        # Voting Ensemble - balanced weights
        self.failure_model = VotingClassifier(
            estimators=[
                ('gb', gb_model),
                ('rf', rf_model),
                ('dt', dt_model)
            ],
            voting='soft',
            weights=[6, 3, 1]
        )
        
        print("\n⏳ Training ensemble (3-4 minutes)...")
        self.failure_model.fit(X_final, y_final)
        print("✅ Training complete!")
        
        # Calculate suggested threshold (but allow manual override)
        y_train_proba = self.failure_model.predict_proba(X_final)[:, 1]
        
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_final, y_train_proba)
        f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
        
        # Suggest threshold with 98%+ recall and best F1
        high_recall_idx = np.where(recall_vals >= 0.98)[0]
        if len(high_recall_idx) > 0:
            best_idx = high_recall_idx[np.argmax(f1_scores[high_recall_idx])]
            self.optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.35
        else:
            best_idx = np.argmax(f1_scores)
            self.optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.40
        
        print(f"\n🎯 Suggested threshold: {self.optimal_threshold:.4f}")
        print(f"   Note: You can manually adjust this in the Threshold Tuning page")
        
        y_train_pred = (y_train_proba >= self.optimal_threshold).astype(int)
        train_cm = confusion_matrix(y_final, y_train_pred)
        tn_tr, fp_tr, fn_tr, tp_tr = train_cm.ravel()
        
        print(f"\n   Training Performance (at suggested threshold):")
        print(f"   Recall: {recall_score(y_final, y_train_pred):.4f}")
        print(f"   F1: {f1_score(y_final, y_train_pred):.4f}")
        print(f"   Precision: {precision_score(y_final, y_train_pred):.4f}")
        print(f"   FN: {fn_tr}, FP: {fp_tr}")
        
        return self.failure_model
    
    def evaluate_model(self, X_test, y_test, custom_threshold=None):
        """
        Evaluate model with optional custom threshold
        If custom_threshold is None, uses the suggested optimal threshold
        """
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        y_pred_proba = self.failure_model.predict_proba(X_test)[:, 1]
        
        # Use custom threshold if provided, otherwise use optimal
        threshold = custom_threshold if custom_threshold is not None else self.optimal_threshold
        
        print(f"\n🎯 Using threshold: {threshold:.4f}")
        if custom_threshold is not None:
            print(f"   (Custom threshold provided)")
        else:
            print(f"   (Suggested optimal threshold)")
        
        # Predictions with threshold
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred) if sum(y_pred) > 0 else 0
        recall = recall_score(y_test, y_pred)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
        
        total_failures = sum(y_test)
        total_normal = len(y_test) - total_failures
        
        print("\n" + "="*70)
        print("📊 PERFORMANCE METRICS")
        print("="*70)
        
        print(f"\n🎯 PRIMARY METRICS:")
        
        # Status indicators
        f1_status = "🎉 OUTSTANDING" if f1>=0.72 else "🎉 EXCELLENT" if f1>=0.68 else "✅ VERY GOOD" if f1>=0.63 else "✅ GOOD" if f1>=0.58 else "⚠️"
        recall_status = "🎉 PERFECT" if recall>=1.0 else "🎉 NEAR-PERFECT" if recall>=0.99 else "✅ EXCELLENT" if recall>=0.97 else "✅ GOOD" if recall>=0.95 else "⚠️"
        prec_status = "🎉 EXCELLENT" if precision>=0.55 else "✅ VERY GOOD" if precision>=0.45 else "✅ GOOD" if precision>=0.38 else "✅ OK" if precision>=0.30 else "⚠️"
        acc_status = "🎉 OUTSTANDING" if accuracy>=0.95 else "✅ EXCELLENT" if accuracy>=0.93 else "✅ GOOD" if accuracy>=0.90 else "⚠️"
        
        print(f"   F1 Score:   {f1:.4f} {f1_status}")
        print(f"   Recall:     {recall:.4f} ({recall*100:.2f}%) {recall_status}")
        print(f"   Precision:  {precision:.4f} ({precision*100:.2f}%) {prec_status}")
        print(f"   Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%) {acc_status}")
        print(f"   ROC-AUC:    {roc_auc:.4f} {'🎉 OUTSTANDING' if roc_auc>=0.98 else '✅ EXCELLENT' if roc_auc>=0.96 else '✅ GOOD'}")
        
        print(f"\n📋 CONFUSION MATRIX:")
        print(f"                      Predicted")
        print(f"                 No Fail  |  Fail")
        print(f"   Actual  No    {tn:6d}   | {fp:5d}")
        print(f"           Fail  {fn:6d}   | {tp:5d}")
        
        print(f"\n🔍 DETAILED ANALYSIS:")
        print(f"   True Negatives:  {tn}")
        
        fp_status = "🎉 OUTSTANDING" if fp<80 else "✅ EXCELLENT" if fp<120 else "✅ VERY GOOD" if fp<160 else "✅ GOOD" if fp<200 else "✅ OK" if fp<250 else "⚠️ HIGH"
        print(f"   False Positives: {fp} (FPR: {fpr*100:.2f}%) {fp_status}")
        
        fn_status = "🎉🎉🎉 PERFECT! ZERO MISSED!" if fn==0 else "✅ OUTSTANDING (1)" if fn==1 else "✅ EXCELLENT (2)" if fn==2 else "⚠️"
        print(f"   False Negatives: {fn} {fn_status}")
        
        print(f"   True Positives:  {tp}")
        
        if total_failures > 0:
            detection_rate = (tp / total_failures) * 100
            miss_rate = (fn / total_failures) * 100
            
            print(f"\n🎯 FAILURE DETECTION:")
            print(f"   Total: {total_failures}")
            print(f"   Detected: {tp} ({detection_rate:.2f}%)")
            print(f"   Missed: {fn} ({miss_rate:.2f}%)")
            
            if fn == 0:
                print(f"\n   🏆🏆🏆 PERFECT! 100% DETECTION! 🏆🏆🏆")
            elif fn <= 2:
                print(f"\n   ✅ EXCELLENT! {detection_rate:.1f}% detection")
        
        print(f"\n💰 COST-BENEFIT ANALYSIS:")
        fa_cost = fp * 100
        mf_cost = fn * 10000
        total_cost = fa_cost + mf_cost
        max_loss = total_failures * 10000
        savings = max_loss - total_cost
        roi = (savings / max(fa_cost, 1)) * 100 if fa_cost > 0 else 999
        
        print(f"   False Alarm Cost:    ${fa_cost:>10,}")
        print(f"   Missed Failure Cost: ${mf_cost:>10,}")
        print(f"   Total Cost:          ${total_cost:>10,}")
        print(f"   Net Savings:         ${savings:>10,}")
        print(f"   ROI:                 {roi:>10,.0f}%")
        
        print(f"\n💡 TIP: Use the Threshold Tuning page to adjust the threshold")
        print(f"   and find the best balance for your use case!")
        
        print("\n" + "="*70)
        print("\n📋 CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, target_names=['No Failure', 'Failure'], digits=4))
        print("="*70)
        
        return {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'specificity': specificity,
            'optimal_threshold': threshold,
            'false_negatives': fn,
            'false_positives': fp,
            'true_positives': tp,
            'true_negatives': tn
        }
    
    def get_threshold_analysis(self, X_test, y_test, threshold_range=None):
        """
        Analyze performance across different thresholds
        Returns metrics for each threshold value
        """
        if threshold_range is None:
            threshold_range = np.linspace(0.01, 0.99, 99)
        
        y_pred_proba = self.failure_model.predict_proba(X_test)[:, 1]
        
        results = []
        for threshold in threshold_range:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            results.append({
                'threshold': threshold,
                'f1': f1_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred) if sum(y_pred) > 0 else 0,
                'recall': recall_score(y_test, y_pred),
                'accuracy': (tp + tn) / (tp + tn + fp + fn),
                'fn': fn,
                'fp': fp,
                'tp': tp,
                'tn': tn
            })
        
        return pd.DataFrame(results)
    
    def predict_failure(self, X, threshold=None):
        """Predict failure with specified threshold"""
        if threshold is None:
            threshold = self.optimal_threshold
        
        probability = self.failure_model.predict_proba(X)[:, 1]
        prediction = (probability >= threshold).astype(int)
        
        return prediction, probability
    
    def save_models(self, failure_path='models/failure_model.pkl', 
                    threshold_path='models/optimal_threshold.pkl'):
        """Save models"""
        joblib.dump(self.failure_model, failure_path)
        joblib.dump(self.optimal_threshold, threshold_path)
        print(f"✓ Models saved (threshold: {self.optimal_threshold:.4f})")
    
    def load_models(self, failure_path='models/failure_model.pkl', 
                    threshold_path='models/optimal_threshold.pkl'):
        """Load models"""
        try:
            self.failure_model = joblib.load(failure_path)
            self.optimal_threshold = joblib.load(threshold_path)
            print(f"✓ Models loaded (threshold: {self.optimal_threshold:.4f})")
        except Exception as e:
            print(f"⚠️ Error loading: {e}")