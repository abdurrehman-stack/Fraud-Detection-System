# backend.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (confusion_matrix, recall_score, accuracy_score, 
                           f1_score, roc_auc_score, precision_recall_curve,
                           classification_report, precision_score)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTETomek
import warnings
from datetime import datetime
import joblib
import os
warnings.filterwarnings('ignore')

class AdvancedFraudDetector:
    def __init__(self):
        self.model = None
        self.threshold = 0.3
        self.feature_columns = None
        self.scaler = RobustScaler()
        self.feature_importance = None
        self.model_metadata = {}
        
    def engineer_features(self, df):
        """Advanced feature engineering"""
        df_copy = df.copy()
        
        # Time-based features
        if 'time_hour' in df_copy.columns:
            df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['time_hour'] / 24)
            df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['time_hour'] / 24)
            df_copy['is_night'] = ((df_copy['time_hour'] >= 22) | (df_copy['time_hour'] <= 5)).astype(int)
            df_copy['is_early_morning'] = ((df_copy['time_hour'] >= 5) & (df_copy['time_hour'] <= 8)).astype(int)
            df_copy['is_business_hours'] = ((df_copy['time_hour'] >= 9) & (df_copy['time_hour'] <= 17)).astype(int)
        
        # Amount-based features
        if 'amount' in df_copy.columns:
            df_copy['amount_log'] = np.log1p(df_copy['amount'])
            df_copy['amount_sqrt'] = np.sqrt(df_copy['amount'])
            df_copy['amount_rank'] = df_copy['amount'].rank(pct=True)
            df_copy['amount_bin_small'] = (df_copy['amount'] < 100).astype(int)
            df_copy['amount_bin_medium'] = ((df_copy['amount'] >= 100) & (df_copy['amount'] < 1000)).astype(int)
            df_copy['amount_bin_large'] = ((df_copy['amount'] >= 1000) & (df_copy['amount'] < 10000)).astype(int)
            df_copy['amount_bin_huge'] = (df_copy['amount'] >= 10000).astype(int)
        
        # Interaction features
        if 'distance_from_home' in df_copy.columns:
            df_copy['amount_per_distance'] = df_copy['amount'] / (df_copy['distance_from_home'] + 1)
            df_copy['log_distance'] = np.log1p(df_copy['distance_from_home'])
        
        # Risk scoring
        if all(col in df_copy.columns for col in ['amount', 'distance_from_home', 'time_hour']):
            df_copy['risk_score'] = (
                (df_copy['amount'] / 1000) * 0.4 +
                (df_copy['distance_from_home'] / 100) * 0.3 +
                (abs(df_copy['time_hour'] - 14) / 12) * 0.3
            )
        
        return df_copy
    
    def prepare_features(self, df):
        """Prepare features with advanced engineering"""
        df_engineered = self.engineer_features(df)
        X = df_engineered.drop('is_fraud', axis=1)
        y = df_engineered['is_fraud']
        self.feature_columns = X.columns.tolist()
        return X, y
    
    def train_ensemble_model(self, X_train, y_train):
        """Train ensemble model with multiple algorithms"""
        fraud_ratio = len(y_train[y_train==0]) / len(y_train[y_train==1])
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        smote_tomek = SMOTETomek(sampling_strategy=0.4, random_state=42)
        X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_scaled, y_train)
        
        xgb_model = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            scale_pos_weight=fraud_ratio, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            use_label_encoder=False, eval_metric='logloss'
        )
        
        lgbm_model = LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.03,
            scale_pos_weight=fraud_ratio, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_split=5,
            min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1
        )
        
        self.model = VotingClassifier(
            estimators=[('xgb', xgb_model), ('lgbm', lgbm_model), ('rf', rf_model)],
            voting='soft', weights=[0.5, 0.3, 0.2]
        )
        
        self.model.fit(X_train_resampled, y_train_resampled)
        
        rf_importance = rf_model.fit(X_train_resampled, y_train_resampled).feature_importances_
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': rf_importance
        }).sort_values('importance', ascending=False)
        
        return self.model
    
    def optimize_threshold(self, y_test, y_prob):
        """Dynamic threshold optimization"""
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
        beta = 2
        f_beta_scores = (1 + beta**2) * (precisions * recalls) / ((beta**2 * precisions) + recalls + 1e-10)
        optimal_idx = np.argmax(f_beta_scores[:-1])
        optimal_threshold = thresholds[optimal_idx]
        high_recall_threshold = thresholds[np.where(recalls[:-1] >= 0.95)[0][0]] if any(recalls[:-1] >= 0.95) else optimal_threshold
        self.threshold = optimal_threshold
        return optimal_threshold, high_recall_threshold
    
    def evaluate(self, X_test, y_test):
        """Enhanced evaluation with multiple metrics"""
        X_test_scaled = self.scaler.transform(X_test)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        optimal_threshold, high_recall_threshold = self.optimize_threshold(y_test, y_prob)
        y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_test_scaled, y_test, cv=skf, scoring='recall')
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_optimal).ravel()
        fraud_cost, false_alarm_cost = 1000, 10
        total_cost = (fn * fraud_cost) + (fp * false_alarm_cost)
        cost_saved = (tp * fraud_cost) - (fp * false_alarm_cost)
        
        results = {
            'recall': recall_score(y_test, y_pred_optimal),
            'recall_high_recall': recall_score(y_test, (y_prob >= high_recall_threshold).astype(int)),
            'precision': precision_score(y_test, y_pred_optimal),
            'accuracy': accuracy_score(y_test, y_pred_optimal),
            'f1': f1_score(y_test, y_pred_optimal),
            'auc': roc_auc_score(y_test, y_prob),
            'threshold_optimal': optimal_threshold,
            'threshold_high_recall': high_recall_threshold,
            'confusion_matrix': confusion_matrix(y_test, y_pred_optimal),
            'y_prob': y_prob,
            'y_pred': y_pred_optimal,
            'cv_recall_mean': cv_scores.mean(),
            'cv_recall_std': cv_scores.std(),
            'total_cost': total_cost,
            'cost_saved': cost_saved,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'classification_report': classification_report(y_test, y_pred_optimal, output_dict=True)
        }
        return results
    
    def predict_batch(self, features_df):
        """Predict multiple transactions"""
        features_engineered = self.engineer_features(features_df)
        for col in self.feature_columns:
            if col not in features_engineered.columns:
                features_engineered[col] = 0
        features_engineered = features_engineered[self.feature_columns]
        features_scaled = self.scaler.transform(features_engineered)
        probabilities = self.model.predict_proba(features_scaled)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)
        
        return pd.DataFrame({
            'probability': probabilities,
            'prediction': predictions,
            'risk': pd.cut(probabilities, bins=[0, 0.3, 0.7, 1], labels=['LOW', 'MEDIUM', 'HIGH'])
        })
    
    def predict_single(self, features_dict):
        """Predict single transaction with explanation"""
        input_df = pd.DataFrame([features_dict])
        input_engineered = self.engineer_features(input_df)
        
        for col in self.feature_columns:
            if col not in input_engineered.columns:
                input_engineered[col] = 0
        
        input_engineered = input_engineered[self.feature_columns]
        input_scaled = self.scaler.transform(input_engineered)
        prob = self.model.predict_proba(input_scaled)[0, 1]
        is_fraud = prob >= self.threshold
        
        # Feature contributions
        feature_contributions = {}
        if hasattr(self.model, 'named_estimators_'):
            for i, feature in enumerate(self.feature_columns):
                if input_engineered[feature].values[0] != 0:
                    importance = self.feature_importance.loc[
                        self.feature_importance['feature'] == feature, 'importance'
                    ].values[0] if len(self.feature_importance.loc[self.feature_importance['feature'] == feature]) > 0 else 0
                    feature_contributions[feature] = abs(input_engineered[feature].values[0] * importance)
            top_features = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)[:5]
        else:
            top_features = []
        
        return {
            'is_fraud': bool(is_fraud),
            'probability': float(prob),
            'risk': 'HIGH' if prob > 0.7 else 'MEDIUM' if prob > 0.3 else 'LOW',
            'threshold_used': float(self.threshold),
            'top_features': top_features,
            'confidence': 'HIGH' if abs(prob - 0.5) > 0.3 else 'MEDIUM' if abs(prob - 0.5) > 0.1 else 'LOW'
        }
    
    def save_model(self, path):
        """Save model and metadata"""
        if not os.path.exists('models'):
            os.makedirs('models')
        model_data = {
            'model': self.model, 'scaler': self.scaler, 'feature_columns': self.feature_columns,
            'threshold': self.threshold, 'feature_importance': self.feature_importance,
            'model_metadata': {'timestamp': datetime.now().isoformat(), 'version': '2.0'}
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path):
        """Load model and metadata"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.threshold = model_data['threshold']
        self.feature_importance = model_data['feature_importance']
        self.model_metadata = model_data.get('model_metadata', {})

# ============ PDF REPORT GENERATOR ============
class PDFReportGenerator:
    def __init__(self, results, detector, feature_importance, training_date):
        self.results = results
        self.detector = detector
        self.feature_importance = feature_importance
        self.training_date = training_date
    
    def generate_report(self):
        """Generate comprehensive PDF report"""
        from io import BytesIO
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
        styles = getSampleStyleSheet()
        story = []
        
        # Styles
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, 
                                    textColor=colors.HexColor('#1f77b4'), spaceAfter=30, alignment=1)
        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=16,
                                      textColor=colors.HexColor('#2c3e50'), spaceAfter=12, spaceBefore=12)
        
        # Title Page
        story.append(Paragraph("Advanced Fraud Detection System", title_style))
        story.append(Paragraph(f"Performance Report", title_style))
        story.append(Spacer(1, 0.5*inch))
        story.append(Paragraph(f"Generated: {self.training_date}", styles['Normal']))
        story.append(PageBreak())
        
        # Executive Summary Table
        story.append(Paragraph("Executive Summary", heading_style))
        metrics_data = [
            ['Metric', 'Value', 'Target', 'Status'],
            ['Recall', f"{self.results['recall']*100:.1f}%", '≥95%', '✅' if self.results['recall'] >= 0.95 else '⚠️'],
            ['Precision', f"{self.results['precision']*100:.1f}%", 'N/A', ''],
            ['F1 Score', f"{self.results['f1']:.3f}", '≥0.8', '✅' if self.results['f1'] >= 0.8 else '⚠️'],
            ['ROC-AUC', f"{self.results['auc']:.3f}", '≥0.95', '✅' if self.results['auc'] >= 0.95 else '⚠️']
        ]
        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(metrics_table)
        story.append(PageBreak())
        
        # Confusion Matrix
        story.append(Paragraph("Confusion Matrix Analysis", heading_style))
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(self.results['confusion_matrix'], annot=True, fmt='d', cmap='RdYlGn_r', ax=ax,
                   xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        story.append(Image(img_buffer, width=4*inch, height=3.5*inch))
        story.append(PageBreak())
        
        # Feature Importance
        story.append(Paragraph("Top 15 Most Important Features", heading_style))
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        top_features = self.feature_importance.head(15)
        ax2.barh(range(len(top_features)), top_features['importance'].values)
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features['feature'].values)
        ax2.set_xlabel('Importance Score'); ax2.invert_yaxis()
        
        img_buffer2 = BytesIO()
        plt.savefig(img_buffer2, format='png', dpi=300, bbox_inches='tight')
        img_buffer2.seek(0)
        plt.close()
        story.append(Image(img_buffer2, width=5*inch, height=4*inch))
        
        doc.build(story)
        buffer.seek(0)
        return buffer