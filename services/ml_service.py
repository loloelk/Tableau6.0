"""
Machine learning service for predicting treatment response.
"""

import pandas as pd
import numpy as np
import os
import joblib
import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
from datetime import datetime
from utils.logging_config import setup_logger
from utils.config_manager import ConfigManager

# Set up logger
logger = setup_logger('ml_service')
config = ConfigManager().get_config()

# Define default feature columns for ML model
DEFAULT_FEATURE_COLUMNS = [
    'age', 'sexe', 'madrs_score_bl', 'hdrs_score_bl', 'duration_episode', 
    'nb_episodes_past', 'treatment_resistance_level', 'anxiety_level',
    'genetic_risk_score', 'sleep_quality', 'stress_level'
]

class MLService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.threshold = 0.5  # Default threshold, will be optimized during training
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'response_prediction_model.pkl')
        self.features_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'selected_features.pkl')
        self.evaluation_cohort_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'evaluation_cohort.pkl')
    
    def generate_evaluation_cohort(self, size=100, balanced=True, save=True):
        """
        Generate a balanced evaluation cohort for model testing
        
        Args:
            size: Number of patients in the cohort
            balanced: Whether to ensure class balance (equal responders and non-responders)
            save: Whether to save the generated cohort
            
        Returns:
            DataFrame containing the evaluation cohort
        """
        try:
            logger.info(f"Generating evaluation cohort of size {size}")
            
            # Load extended patient data which has all potential features
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    'data', 'extended_patient_data_ml.csv')
            
            if not os.path.exists(data_path):
                logger.error(f"Data file not found: {data_path}")
                return pd.DataFrame()
                
            df = pd.read_csv(data_path)
            
            # Validate data has required columns
            if 'is_responder' not in df.columns:
                logger.error("Column 'is_responder' missing in dataset")
                return df.sample(min(10, len(df)), random_state=42)
            
            # Ensure necessary columns exist
            for col in ['madrs_score_bl', 'madrs_score_fu']:
                if col not in df.columns:
                    logger.warning(f"Column '{col}' missing in dataset")
            
            # Split between responders and non-responders
            responders = df[df['is_responder'] == 1]
            non_responders = df[df['is_responder'] == 0]
            
            logger.info(f"Found {len(responders)} responders and {len(non_responders)} non-responders in dataset")
            
            if balanced:
                # Select equal numbers from each class
                n_per_class = min(size // 2, min(len(responders), len(non_responders)))
                responders_sample = responders.sample(n=n_per_class, random_state=42)
                non_responders_sample = non_responders.sample(n=n_per_class, random_state=42)
                cohort = pd.concat([responders_sample, non_responders_sample])
                
                # Shuffle the final cohort
                cohort = cohort.sample(frac=1, random_state=42).reset_index(drop=True)
                
                logger.info(f"Created balanced evaluation cohort with {len(cohort)} patients "
                           f"({len(responders_sample)} responders, {len(non_responders_sample)} non-responders)")
            else:
                # Use random sampling with the original class distribution
                cohort = df.sample(n=min(size, len(df)), random_state=42)
                logger.info(f"Created evaluation cohort with {len(cohort)} patients "
                           f"({sum(cohort['is_responder'])} responders, {len(cohort) - sum(cohort['is_responder'])} non-responders)")
            
            if save:
                with open(self.evaluation_cohort_path, 'wb') as f:
                    pickle.dump(cohort, f)
                logger.info(f"Saved evaluation cohort to {self.evaluation_cohort_path}")
            
            return cohort
            
        except Exception as e:
            logger.error(f"Error generating evaluation cohort: {e}")
            # Return a small default cohort in case of error
            return df.sample(n=min(10, len(df)), random_state=42)
    
    def get_evaluation_cohort(self):
        """
        Load the saved evaluation cohort or generate a new one if it doesn't exist
        
        Returns:
            DataFrame containing the evaluation cohort
        """
        try:
            if os.path.exists(self.evaluation_cohort_path):
                with open(self.evaluation_cohort_path, 'rb') as f:
                    cohort = pickle.load(f)
                logger.info(f"Loaded evaluation cohort with {len(cohort)} patients")
                return cohort
            else:
                logger.info("No evaluation cohort found, generating new one")
                return self.generate_evaluation_cohort()
        except Exception as e:
            logger.error(f"Error loading evaluation cohort: {e}")
            return self.generate_evaluation_cohort()
    
    def train_response_prediction_model(self, save_model=True):
        """
        Train a machine learning model to predict treatment response
        
        Args:
            save_model: Whether to save the trained model
            
        Returns:
            Trained model and evaluation metrics
        """
        try:
            logger.info("Training response prediction model")
            
            # Load training data
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'ml_training_data.csv')
            df = pd.read_csv(data_path)
            
            # Check class balance
            n_responders = sum(df['is_responder'])
            n_non_responders = len(df) - n_responders
            response_rate = n_responders / len(df)
            
            logger.info(f"Training data: {len(df)} patients, {n_responders} responders ({response_rate:.1%}), "
                       f"{n_non_responders} non-responders ({1-response_rate:.1%})")
            
            # Define features and target
            X = df.drop(['patient_id', 'is_responder', 'madrs_score_fu'], axis=1)
            y = df['is_responder']
            
            # Handle categorical features
            categorical_features = ['sexe']
            numeric_features = [col for col in X.columns if col not in categorical_features]
            
            # Create preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
                ])
            
            # Create model pipeline with feature selection
            pipeline = Pipeline([
                ('preprocess', preprocessor),
                ('select', SelectKBest(f_classif, k=min(10, len(numeric_features) + len(categorical_features) - 1))),
                ('classifier', CalibratedClassifierCV(
                    estimator=RandomForestClassifier(random_state=42),
                    cv=5,
                    method='sigmoid'
                ))
            ])
            
            # Define parameter grid for hyperparameter tuning
            param_grid = {
                'classifier__estimator__n_estimators': [50, 100, 200],
                'classifier__estimator__max_depth': [None, 5, 10],
                'classifier__estimator__min_samples_split': [2, 5],
                'classifier__estimator__min_samples_leaf': [1, 2, 4],
                'classifier__estimator__class_weight': [None, 'balanced']
            }
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            # Use grid search for hyperparameter tuning
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=5,
                scoring='recall',  # Optimize for recall (sensitivity) to address the 0 sensitivity issue
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
            
            # Get the selected features from the pipeline
            preprocessed_X_train = best_model.named_steps['preprocess'].transform(X_train)
            selector = best_model.named_steps['select']
            selected_indices = selector.get_support(indices=True)
            
            # Get feature names after one-hot encoding
            preprocessor = best_model.named_steps['preprocess']
            feature_names_out = (
                numeric_features +
                [f"{col}_{cat}" for col in categorical_features for cat in preprocessor.transformers_[1][1].categories_[0][1:]]
            )
            
            selected_features = [feature_names_out[i] for i in selected_indices]
            logger.info(f"Selected features: {selected_features}")
            
            # Get predictions on test set
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            logger.info(f"Model evaluation metrics (default threshold 0.5):")
            logger.info(f"Accuracy: {accuracy:.2f}")
            logger.info(f"Precision: {precision:.2f}")
            logger.info(f"Recall (sensitivity): {recall:.2f}")
            logger.info(f"F1 score: {f1:.2f}")
            logger.info(f"ROC AUC: {roc_auc:.2f}")
            
            # Optimize threshold for better sensitivity/specificity balance
            thresholds = np.linspace(0.1, 0.9, 9)
            metrics = {}
            
            for threshold in thresholds:
                y_pred_threshold = (y_pred_proba >= threshold).astype(int)
                
                # Calculate metrics for this threshold
                precision_t = precision_score(y_test, y_pred_threshold)
                recall_t = recall_score(y_test, y_pred_threshold)
                f1_t = f1_score(y_test, y_pred_threshold)
                
                # Store metrics
                metrics[threshold] = {
                    'precision': precision_t,
                    'recall': recall_t,
                    'f1': f1_t,
                    'balanced_score': (precision_t + recall_t) / 2
                }
            
            # Select optimal threshold based on balanced precision and recall
            optimal_threshold = max(metrics.keys(), key=lambda x: metrics[x]['balanced_score'])
            optimal_metrics = metrics[optimal_threshold]
            
            logger.info(f"Optimal threshold: {optimal_threshold:.2f}")
            logger.info(f"Metrics at optimal threshold:")
            logger.info(f"Precision: {optimal_metrics['precision']:.2f}")
            logger.info(f"Recall (sensitivity): {optimal_metrics['recall']:.2f}")
            logger.info(f"F1 score: {optimal_metrics['f1']:.2f}")
            
            # Save the threshold
            self.threshold = optimal_threshold
            
            # Save the model and selected features if requested
            if save_model:
                # Save model
                joblib.dump(best_model, self.model_path)
                
                # Save selected features
                with open(self.features_path, 'wb') as f:
                    pickle.dump({
                        'features': selected_features,
                        'threshold': optimal_threshold
                    }, f)
                
                logger.info(f"Saved model to {self.model_path}")
                logger.info(f"Saved selected features to {self.features_path}")
            
            # Store the model and selected features
            self.model = best_model
            self.selected_features = selected_features
            
            # Evaluate on an independent evaluation cohort
            cohort = self.get_evaluation_cohort()

            # Validate required columns
            required_cols = ['is_responder']
            missing_cols = [col for col in required_cols if col not in cohort.columns]
            if missing_cols:
                logger.error(f"Missing required columns in evaluation cohort: {missing_cols}")
                return {'error': f"Missing required columns: {missing_cols}", 'metrics': {}}

            # Check for empty values
            empty_cols = [col for col in required_cols 
                        if col in cohort.columns and cohort[col].isna().all()]
            if empty_cols:
                logger.error(f"Required columns have all empty values: {empty_cols}")
                return {'error': f"Empty values in required columns: {empty_cols}", 'metrics': {}}

            X_cohort = cohort.drop(['patient_id', 'is_responder', 'madrs_score_fu'], axis=1, errors='ignore')
            y_cohort = cohort['is_responder']
            
            cohort_proba = best_model.predict_proba(X_cohort)[:, 1]
            cohort_pred = (cohort_proba >= optimal_threshold).astype(int)
            
            # Calculate cohort metrics
            cohort_accuracy = accuracy_score(y_cohort, cohort_pred)
            cohort_precision = precision_score(y_cohort, cohort_pred)
            cohort_recall = recall_score(y_cohort, cohort_pred)
            cohort_f1 = f1_score(y_cohort, cohort_pred)
            cohort_roc_auc = roc_auc_score(y_cohort, cohort_proba)
            
            logger.info(f"Evaluation cohort metrics (threshold {optimal_threshold:.2f}):")
            logger.info(f"Accuracy: {cohort_accuracy:.2f}")
            logger.info(f"Precision: {cohort_precision:.2f}")
            logger.info(f"Recall (sensitivity): {cohort_recall:.2f}")
            logger.info(f"F1 score: {cohort_f1:.2f}")
            logger.info(f"ROC AUC: {cohort_roc_auc:.2f}")
            
            return {
                'model': best_model,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'threshold': optimal_threshold,
                    'optimal_precision': optimal_metrics['precision'],
                    'optimal_recall': optimal_metrics['recall'],
                    'optimal_f1': optimal_metrics['f1']
                },
                'cohort_metrics': {
                    'accuracy': cohort_accuracy,
                    'precision': cohort_precision,
                    'recall': cohort_recall,
                    'f1': cohort_f1,
                    'roc_auc': cohort_roc_auc
                }
            }
        
        except Exception as e:
            logger.error(f"Error training model: {e}")
            logger.exception("Exception details:")
            return None
    
    def load_model(self):
        """
        Load the trained model and selected features
        
        Returns:
            Loaded model or None if loading fails
        """
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.features_path):
                # Load model
                self.model = joblib.load(self.model_path)
                
                # Load selected features and threshold
                with open(self.features_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.selected_features = saved_data['features']
                    self.threshold = saved_data.get('threshold', 0.5)  # Default to 0.5 if not found
                
                logger.info(f"Loaded model from {self.model_path}")
                logger.info(f"Loaded {len(self.selected_features)} selected features and threshold {self.threshold}")
                return self.model
            else:
                logger.warning("Model or feature files not found. Training new model.")
                result = self.train_response_prediction_model()
                return result['model'] if result else None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.exception("Exception details:")
            return None
    
    def predict_response_probability(self, patient_data):
        """
        Predict the probability of response for a new patient
        
        Args:
            patient_data: DataFrame, Series, or dict containing patient data
            
        Returns:
            Dictionary with response probability and classification
        """
        try:
            # Handle different input types
            if isinstance(patient_data, dict):
                patient_data = pd.DataFrame([patient_data])
            elif isinstance(patient_data, pd.Series):
                patient_data = patient_data.to_frame().T
            elif isinstance(patient_data, pd.DataFrame):
                # Ensure we're working with a single row
                if len(patient_data) > 1:
                    logger.warning(f"Received multiple rows ({len(patient_data)}), using first row")
                    patient_data = patient_data.iloc[[0]]
            
            # Check if selected features are defined and available
            if self.selected_features is not None:
                missing_features = [f for f in self.selected_features if f not in patient_data.columns]
                if missing_features:
                    logger.error(f"Missing required features: {missing_features}")
                    return {
                        'probability': 0.5,
                        'is_responder': False,
                        'confidence': 'low',
                        'error': f'Missing features: {missing_features}'
                    }
            
            # Load model if not loaded
            if self.model is None:
                self.load_model()
            
            if self.model is None:
                logger.error("Could not load or train model for prediction")
                return {
                    'probability': 0.5,
                    'is_responder': False,
                    'confidence': 'low',
                    'error': 'Model not available'
                }
            
            # Make prediction
            response_prob = self.model.predict_proba(patient_data)[:, 1][0]
            
            # Apply optimized threshold
            is_responder = response_prob >= self.threshold
            
            # Determine confidence level
            if abs(response_prob - 0.5) > 0.3:
                confidence = 'high'
            elif abs(response_prob - 0.5) > 0.15:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            logger.info(f"Prediction for patient: probability={response_prob:.2f}, "
                      f"is_responder={is_responder}, confidence={confidence}")
            
            return {
                'probability': float(response_prob),  # Convert numpy float to Python float for JSON serialization
                'is_responder': bool(is_responder),   # Convert numpy bool to Python bool
                'confidence': confidence,
                'threshold_used': float(self.threshold)
            }
        
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            logger.exception("Exception details:")
            return {
                'probability': 0.5,
                'is_responder': False,
                'confidence': 'low',
                'error': str(e)
            }
    
    def get_model_performance(self):
        """
        Get performance metrics for the current model
        
        Returns:
            Dictionary with model performance metrics
        """
        try:
            # Load model if not loaded
            if self.model is None:
                self.load_model()
            
            if self.model is None:
                logger.error("Could not load or train model for performance evaluation")
                return {
                    'error': 'Model not available',
                    'metrics': {}
                }
            
            # Get evaluation cohort
            cohort = self.get_evaluation_cohort()
            
            # Validate required columns
            required_cols = ['is_responder']
            missing_cols = [col for col in required_cols if col not in cohort.columns]
            if missing_cols:
                logger.error(f"Missing required columns in evaluation cohort: {missing_cols}")
                return {'error': f"Missing required columns: {missing_cols}", 'metrics': {}}

            # Check for empty values
            empty_cols = [col for col in required_cols 
                         if col in cohort.columns and cohort[col].isna().all()]
            if empty_cols:
                logger.error(f"Required columns have all empty values: {empty_cols}")
                return {'error': f"Empty values in required columns: {empty_cols}", 'metrics': {}}
            
            # Prepare data
            X_cohort = cohort.drop(['patient_id', 'is_responder', 'madrs_score_fu'], axis=1, errors='ignore')
            y_cohort = cohort['is_responder']
            
            # Make predictions
            y_pred_proba = self.model.predict_proba(X_cohort)[:, 1]
            
            # Evaluate at default threshold
            y_pred_default = (y_pred_proba >= 0.5).astype(int)
            
            # Evaluate at optimized threshold
            y_pred_optimized = (y_pred_proba >= self.threshold).astype(int)
            
            # Calculate metrics
            metrics_default = {
                'accuracy': accuracy_score(y_cohort, y_pred_default),
                'precision': precision_score(y_cohort, y_pred_default),
                'recall': recall_score(y_cohort, y_pred_default),
                'f1': f1_score(y_cohort, y_pred_default),
                'threshold': 0.5
            }
            
            metrics_optimized = {
                'accuracy': accuracy_score(y_cohort, y_pred_optimized),
                'precision': precision_score(y_cohort, y_pred_optimized),
                'recall': recall_score(y_cohort, y_pred_optimized),
                'f1': f1_score(y_cohort, y_pred_optimized),
                'threshold': self.threshold
            }
            
            # Calculate ROC AUC
            roc_auc = roc_auc_score(y_cohort, y_pred_proba)
            
            # Get thresholds and corresponding metrics for ROC curve
            thresholds = np.linspace(0, 1, 101)[1:-1]  # 99 thresholds between 0.01 and 0.99
            threshold_metrics = []
            
            for threshold in thresholds:
                y_pred_t = (y_pred_proba >= threshold).astype(int)
                precision = precision_score(y_cohort, y_pred_t, zero_division=0)
                recall = recall_score(y_cohort, y_pred_t, zero_division=0)
                f1 = f1_score(y_cohort, y_pred_t, zero_division=0)
                
                threshold_metrics.append({
                    'threshold': float(threshold),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1': float(f1)
                })
            
            # Calculate confusion matrix at optimized threshold
            TP = ((y_cohort == 1) & (y_pred_optimized == 1)).sum()
            FP = ((y_cohort == 0) & (y_pred_optimized == 1)).sum()
            TN = ((y_cohort == 0) & (y_pred_optimized == 0)).sum()
            FN = ((y_cohort == 1) & (y_pred_optimized == 0)).sum()
            
            confusion_matrix = {
                'true_positive': int(TP),
                'false_positive': int(FP),
                'true_negative': int(TN),
                'false_negative': int(FN)
            }
            
            logger.info(f"Model performance on evaluation cohort:")
            logger.info(f"Default threshold (0.5): "
                      f"accuracy={metrics_default['accuracy']:.2f}, "
                      f"precision={metrics_default['precision']:.2f}, "
                      f"recall={metrics_default['recall']:.2f}, "
                      f"f1={metrics_default['f1']:.2f}")
            
            logger.info(f"Optimized threshold ({self.threshold:.2f}): "
                      f"accuracy={metrics_optimized['accuracy']:.2f}, "
                      f"precision={metrics_optimized['precision']:.2f}, "
                      f"recall={metrics_optimized['recall']:.2f}, "
                      f"f1={metrics_optimized['f1']:.2f}")
            
            return {
                'metrics_default': metrics_default,
                'metrics_optimized': metrics_optimized,
                'roc_auc': float(roc_auc),
                'threshold_metrics': threshold_metrics,
                'confusion_matrix': confusion_matrix,
                'cohort_size': len(cohort),
                'response_rate': float(y_cohort.mean()),
                'feature_importance': self.get_feature_importance()
            }
        
        except Exception as e:
            logger.error(f"Error evaluating model performance: {e}")
            logger.exception("Exception details:")
            return {
                'error': str(e),
                'metrics': {}
            }
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model
        
        Returns:
            Dictionary mapping feature names to their importance scores
        """
        try:
            # Load model if not loaded
            if self.model is None:
                self.load_model()
            
            if self.model is None or self.selected_features is None:
                logger.error("Could not load model or selected features")
                return {}
            
            # Check if model has feature_importances_ attribute (tree-based models)
            if hasattr(self.model.named_steps['classifier'].estimator, 'feature_importances_'):
                # Get feature importances from the model
                classifier = self.model.named_steps['classifier']
                
                # For CalibratedClassifierCV, get the base estimator
                if hasattr(classifier, 'estimators_'):
                    base_model = classifier.estimators_[0].base_estimator
                else:
                    base_model = classifier.estimator
                
                importances = base_model.feature_importances_
                
                # Sort features by importance
                feature_importances = dict(zip(self.selected_features, importances))
                sorted_importances = {k: float(v) for k, v in sorted(
                    feature_importances.items(), 
                    key=lambda item: item[1], 
                    reverse=True
                )}
                
                return sorted_importances
            else:
                # For models without direct feature_importances_
                logger.warning("Model does not provide feature importances")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            logger.exception("Exception details:")
            return {}

# Standalone function to be imported directly from the ml_service module
def predict_response_probability(patient_data):
    """
    Standalone function to predict response probability for a patient.
    This wraps the MLService class method for easier importing.
    
    Args:
        patient_data: DataFrame or dict containing patient data
        
    Returns:
        Dictionary with response probability and classification
    """
    try:
        service = MLService()
        # The class method already returns a properly formatted dictionary
        return service.predict_response_probability(patient_data)
    except Exception as e:
        import logging
        logger = logging.getLogger('ml_service')
        logger.error(f"Error in standalone predict_response_probability: {e}")
        # Return a standardized error response
        return {
            'probability': 0.5,
            'is_responder': False,
            'confidence': 'low',
            'error': str(e)
        }

# Add a standalone generate_evaluation_cohort function for easier importing
def generate_evaluation_cohort(size=100, balanced=True, save=True):
    """
    Standalone function to generate an evaluation cohort.
    This wraps the MLService class method for easier importing.
    
    Args:
        size: Number of patients in the cohort
        balanced: Whether to ensure class balance
        save: Whether to save the generated cohort
        
    Returns:
        DataFrame containing the evaluation cohort
    """
    service = MLService()
    return service.generate_evaluation_cohort(size=size, balanced=balanced, save=save)

# Make sure all functions and constants are available at the module level for imports
__all__ = ['MLService', 'predict_response_probability', 'generate_evaluation_cohort', 'DEFAULT_FEATURE_COLUMNS']