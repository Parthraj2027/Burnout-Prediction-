import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_fscore_support, log_loss)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

from config.config import MODEL_PARAMS, SMOTE_PARAMS

class BurnoutPredictor:
    def __init__(self, approach='merged'):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        self.training_history = {
            'train_loss': [],
            'val_loss': []
        }

    def train(self, X_train, y_train, X_val=None, y_val=None, use_custom_loss=True):
        print("\n" + "="*60)
        print("TRAINING MODEL WITH VALIDATION")
        print("="*60)

        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()

        print("\nBalancing classes with SMOTE...")
        smote = SMOTE(**SMOTE_PARAMS)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        print(f"  Original samples: {len(y_train)}")
        print(f"  After SMOTE: {len(y_train_balanced)}")
        print(f"  Class distribution after SMOTE:")
        print(f"    No Burnout (0): {sum(y_train_balanced == 0)}")
        print(f"    Burnout (1): {sum(y_train_balanced == 1)}")

        scale_pos_weight = len(y_train_balanced[y_train_balanced == 0]) / len(y_train_balanced[y_train_balanced == 1])
        print(f"  Scale positive weight: {scale_pos_weight:.2f}")

        model_config = MODEL_PARAMS.copy()
        model_config['scale_pos_weight'] = scale_pos_weight

        if use_custom_loss:
            print("\nUsing Weighted XGBoost with enhanced class balancing...")

            model_config['scale_pos_weight'] = scale_pos_weight * 1.5
        else:
            print("\nUsing Standard XGBoost...")

        print("\nTraining XGBoost classifier...")
        self.model = XGBClassifier(**model_config)

        eval_set = [(X_train_balanced, y_train_balanced)]

        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            print(f"  Validation set: {len(X_val)} samples")

        if X_val is not None:
            self.model.fit(
                X_train_balanced, y_train_balanced,
                eval_set=eval_set,
                verbose=False
            )

            results = self.model.evals_result()
            if 'validation_0' in results:
                self.training_history['train_loss'] = results['validation_0'].get('logloss', [])
            if 'validation_1' in results:
                self.training_history['val_loss'] = results['validation_1'].get('logloss', [])

            print(f"\n  Training completed: {self.model.n_estimators} iterations")
            if self.training_history['val_loss']:
                final_val_loss = self.training_history['val_loss'][-1]
                print(f"  Final validation loss: {final_val_loss:.4f}")
        else:
            self.model.fit(X_train_balanced, y_train_balanced)

        self.is_trained = True

        print("✓ Model training complete!")
        print("="*60)

        return self.model

    def plot_training_history(self, save_path='./models/training_history.png'):
        if not self.training_history['train_loss']:
            print("No training history available")
            return

        plt.figure(figsize=(10, 6))

        iterations = range(1, len(self.training_history['train_loss']) + 1)
        plt.plot(iterations, self.training_history['train_loss'],
                label='Training Loss', linewidth=2, color='#2E86AB', alpha=0.8)
        has_labels = True

        if self.training_history['val_loss']:
            plt.plot(iterations, self.training_history['val_loss'],
                    label='Validation Loss', linewidth=2, color='#A23B72', alpha=0.8)

            best_iter = np.argmin(self.training_history['val_loss']) + 1
            best_loss = min(self.training_history['val_loss'])
            plt.axvline(x=best_iter, color='green', linestyle='--',
                       linewidth=1.5, alpha=0.6, label=f'Best Iteration ({best_iter})')
            plt.scatter([best_iter], [best_loss], color='green', s=100,
                       zorder=5, marker='*', edgecolors='darkgreen', linewidth=1.5)

        plt.xlabel('Iteration', fontsize=12, fontweight='bold')
        plt.ylabel('Log Loss', fontsize=12, fontweight='bold')
        plt.title('Training History - Model Learning Curve', fontsize=14, fontweight='bold', pad=15)

        if has_labels:
            plt.legend(loc='best', fontsize=10, framealpha=0.9)

        plt.grid(True, alpha=0.3, linestyle='--')

        if self.training_history['val_loss']:
            stats_text = f'Final Train Loss: {self.training_history["train_loss"][-1]:.4f}\n'
            stats_text += f'Final Val Loss: {self.training_history["val_loss"][-1]:.4f}\n'
            stats_text += f'Best Val Loss: {best_loss:.4f}'
            plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nTraining history plot saved: {save_path}")
        plt.close()

    def predict(self, X):
        if not self.is_trained:
            raise ValueError("Model is not trained yet! Call train() first.")

        predictions = self.model.predict(X)
        return predictions

    def predict_proba(self, X):
        if not self.is_trained:
            raise ValueError("Model is not trained yet! Call train() first.")

        probabilities = self.model.predict_proba(X)[:, 1]
        return probabilities

    def predict_with_threshold(self, X, threshold=0.4):
        probabilities = self.predict_proba(X)
        predictions = (probabilities >= threshold).astype(int)
        return predictions

    def predict_single(self, features_dict):
        if not self.is_trained:
            raise ValueError("Model is not trained yet! Call train() first.")

        df = pd.DataFrame([features_dict])

        if self.feature_names:
            for feat in self.feature_names:
                if feat not in df.columns:
                    df[feat] = 0
            df = df[self.feature_names]

        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0, 1]

        return prediction, probability

    def check_accuracy(self, X_test, y_test):
        print("\n" + "="*60)
        print("CHECKING MODEL ACCURACY")
        print("="*60)

        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )

        print(f"\n📊 OVERALL ACCURACY: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"📈 ROC-AUC Score: {roc_auc:.4f}")
        print(f"🎯 Precision: {precision:.4f}")
        print(f"🔍 Recall: {recall:.4f}")
        print(f"⚖️  F1-Score: {f1:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        print("\n📋 CONFUSION MATRIX:")
        print("                  Predicted")
        print("                No Burnout  Burnout")
        print(f"Actual No Burnout    {cm[0,0]:6d}    {cm[0,1]:6d}")
        print(f"Actual Burnout       {cm[1,0]:6d}    {cm[1,1]:6d}")

        print(f"\n✓ True Negatives (Correctly predicted No Burnout): {cm[0,0]}")
        print(f"✗ False Positives (Incorrectly predicted Burnout): {cm[0,1]}")
        print(f"✗ False Negatives (Missed Burnout cases): {cm[1,0]}")
        print(f"✓ True Positives (Correctly predicted Burnout): {cm[1,1]}")

        print("\n📝 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred,
                                   target_names=['No Burnout', 'Burnout']))

        print("="*60)

        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

    def save_model(self, filename='burnout_model.pkl'):
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filename)
        print(f"\n✓ Model saved to '{filename}'")

    def load_model(self, filename='burnout_model.pkl'):
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.scaler = model_data.get('scaler')
        self.feature_names = model_data.get('feature_names')
        self.is_trained = True
        print(f"\n✓ Model loaded from '{filename}'")
