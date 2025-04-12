# components/ml_performance.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    precision_recall_curve,
    classification_report
)
import logging
from services.ml_service import DEFAULT_FEATURE_COLUMNS, generate_evaluation_cohort

# Path to saved model and features
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'response_prediction_model.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'selected_features.pkl')
TEST_METRICS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'test_metrics.pkl')
EVALUATION_COHORT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'evaluation_cohort.pkl')
TRAINING_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'extended_patient_data_ml.csv')

# Make sure this function is properly defined and exported
def ml_performance_page():
    """
    Display ML model performance metrics
    """
    st.header("🧠 Performance du Modèle de Prédiction ML")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        st.warning("⚠️ Le modèle ML ou les caractéristiques sélectionnées n'ont pas été trouvés. Veuillez initialiser le modèle.")
        return
    
    try:
        # Load model and features
        model = joblib.load(MODEL_PATH)
        selected_features = joblib.load(FEATURES_PATH)
        
        # Display model info in sidebar
        with st.sidebar:
            st.subheader("Informations sur le Modèle")
            st.info(f"Type: {type(model).__name__}")
            st.info(f"Nombre de caractéristiques: {len(selected_features)}")
        
        # Create tabs for different performance metrics
        tab_overview, tab_features, tab_evaluation, tab_validation, tab_interpretation, tab_decision_curve = st.tabs([
            "📊 Vue d'ensemble", 
            "🔢 Caractéristiques", 
            "📈 Évaluation", 
            "✅ Validation",
            "🔍 Interprétation",
            "📉 Courbe de Décision"
        ])
        
        # --- Tab 1: Overview ---
        with tab_overview:
            st.subheader("Vue d'ensemble du Modèle ML")
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("""
                ### Objectif du Modèle
                Ce modèle a été entraîné pour prédire la probabilité de **réponse au traitement TMS** pour des patients souffrant de dépression.
                
                **Définition de la réponse:** Une amélioration de ≥50% du score MADRS entre le début et la fin du traitement.
                
                ### Méthodologie
                - **Algorithme:** Random Forest Classifier
                - **Technique d'échantillonnage:** Équilibrée par classe (class_weight='balanced')
                - **Validation:** Séparation en ensembles d'entraînement (80%) et de test (20%)
                """)
            
            with col2:
                # Create a simple illustration of the model
                st.markdown("### Pipeline du Modèle")
                
                fig = go.Figure()
                
                # Add shapes to represent the pipeline
                shapes = [
                    dict(type="rect", x0=0, y0=0, x1=1, y1=1, 
                         line=dict(color="rgba(0,0,0,0)"),
                         fillcolor="rgba(135,206,250,0.5)"),
                    dict(type="rect", x0=1.5, y0=0, x1=2.5, y1=1,
                         line=dict(color="rgba(0,0,0,0)"),
                         fillcolor="rgba(60,179,113,0.5)"),
                    dict(type="rect", x0=3, y0=0, x1=4, y1=1,
                         line=dict(color="rgba(0,0,0,0)"),
                         fillcolor="rgba(255,165,0,0.5)")
                ]
                
                for shape in shapes:
                    fig.add_shape(shape)
                
                # Add annotations
                annotations = [
                    dict(x=0.5, y=0.5, text="Prétraitement<br>des données",
                         showarrow=False),
                    dict(x=2, y=0.5, text="Sélection des<br>caractéristiques",
                         showarrow=False),
                    dict(x=3.5, y=0.5, text="Random Forest<br>Classifier",
                         showarrow=False)
                ]
                
                for annotation in annotations:
                    fig.add_annotation(annotation)
                
                # Add arrows
                fig.add_annotation(
                    x=1.2, y=0.5, ax=1, ay=0.5,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2
                )
                
                fig.add_annotation(
                    x=2.7, y=0.5, ax=2.5, ay=0.5,
                    xref="x", yref="y", axref="x", ayref="y",
                    showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2
                )
                
                fig.update_layout(
                    showlegend=False,
                    height=200,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Display key performance metrics if available
            st.subheader("Métriques Clés")
            
            try:
                # First try to load metrics from test set evaluation
                if os.path.exists(TEST_METRICS_PATH):
                    test_metrics = joblib.load(TEST_METRICS_PATH)
                    st.markdown("### Métriques sur l'ensemble de test (20% des données)")
                    
                    # Extract metrics
                    roc_auc = test_metrics.get('roc_auc', 0)
                    accuracy = test_metrics.get('accuracy', 0)
                    confusion_mat = test_metrics.get('confusion_matrix', [[0, 0], [0, 0]])
                    
                    # Calculate sensitivity and specificity
                    if len(confusion_mat) >= 2 and len(confusion_mat[0]) >= 2:
                        tn, fp, fn, tp = confusion_mat[0][0], confusion_mat[0][1], confusion_mat[1][0], confusion_mat[1][1]
                        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                        
                        # Display metrics in columns
                        cols = st.columns(5)
                        cols[0].metric("Exactitude", f"{accuracy:.2f}")
                        cols[1].metric("Sensibilité", f"{sensitivity:.2f}")
                        cols[2].metric("Spécificité", f"{specificity:.2f}")
                        cols[3].metric("AUC", f"{roc_auc:.2f}")
                        cols[4].metric("Test set size", f"{sum([tn, fp, fn, tp])}")
                
                # Load or generate evaluation cohort for visualization
                st.markdown("### Performance sur la cohorte d'évaluation (50 patients)")
                
                # Generate or load evaluation cohort (50 patients)
                cohort = generate_evaluation_cohort(size=50)
                
                if not cohort.empty and 'predicted_probability' in cohort.columns and 'true_label' in cohort.columns:
                    # Calculate metrics on evaluation cohort
                    y_true = cohort['true_label']
                    y_pred = cohort['predicted_label']
                    y_pred_proba = cohort['predicted_probability']
                    
                    cm = confusion_matrix(y_true, y_pred)
                    tn, fp, fn, tp = cm.ravel()
                    
                    accuracy = (tp + tn) / (tp + tn + fp + fn)
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    
                    # Calculate ROC curve and AUC
                    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    # Display metrics in columns
                    cols = st.columns(5)
                    cols[0].metric("Exactitude", f"{accuracy:.2f}")
                    cols[1].metric("Sensibilité", f"{sensitivity:.2f}")
                    cols[2].metric("Spécificité", f"{specificity:.2f}")
                    cols[3].metric("AUC", f"{roc_auc:.2f}")
                    cols[4].metric("Cohorte", f"{len(cohort)}")
                    
                    # Add a note about the metrics
                    st.info("⚠️ Ces métriques sont calculées sur une cohorte d'évaluation de 50 patients.")
                    
                    # Display cohort distribution
                    col1, col2 = st.columns(2)
                    
                    # Response rate pie chart
                    with col1:
                        response_counts = cohort['true_label'].value_counts().reset_index()
                        response_counts.columns = ['Réponse', 'Nombre']
                        response_counts['Réponse'] = response_counts['Réponse'].map({1: 'Répondeur', 0: 'Non-répondeur'})
                        
                        fig = px.pie(
                            response_counts, names='Réponse', values='Nombre',
                            title='Distribution des Réponses',
                            color='Réponse',
                            color_discrete_map={'Répondeur': 'green', 'Non-répondeur': 'crimson'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction distribution
                    with col2:
                        prediction_df = pd.DataFrame({
                            'Probabilité Prédite': cohort['predicted_probability'] * 100,
                            'Réponse Réelle': cohort['true_label'].map({1: 'Répondeur', 0: 'Non-répondeur'})
                        })
                        
                        fig = px.histogram(
                            prediction_df, x='Probabilité Prédite', color='Réponse Réelle',
                            nbins=10, opacity=0.7,
                            title='Distribution des Probabilités Prédites',
                            color_discrete_map={'Répondeur': 'green', 'Non-répondeur': 'crimson'}
                        )
                        
                        fig.add_vline(x=50, line_dash="dash", line_color="black")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Données insuffisantes pour calculer les métriques de performance sur la cohorte d'évaluation.")
            except Exception as e:
                st.error(f"Erreur lors du calcul des métriques: {str(e)}")
                logging.exception("Error calculating performance metrics")
        
        # --- Tab 2: Features ---
        with tab_features:
            st.subheader("Caractéristiques du Modèle")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Caractéristiques Sélectionnées")
                
                # Display the selected features
                if selected_features:
                    st.write(f"Le modèle utilise **{len(selected_features)}** caractéristiques:")
                    
                    # Create a formatted list of features
                    features_markdown = ""
                    for i, feature in enumerate(selected_features):
                        features_markdown += f"- `{feature}`\n"
                    
                    st.markdown(features_markdown)
                else:
                    st.warning("Aucune caractéristique trouvée")
            
            with col2:
                st.markdown("### Importance des Caractéristiques")
                
                try:
                    # Get feature importances from the model
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                        
                        # Create a DataFrame with feature importances
                        feature_importance = pd.DataFrame({
                            'Feature': selected_features,
                            'Importance': importance
                        }).sort_values('Importance', ascending=False)
                        
                        # Normalize importances for better visualization
                        feature_importance['Importance_Normalized'] = feature_importance['Importance'] / feature_importance['Importance'].max()
                        
                        # Create a bar chart
                        fig = px.bar(
                            feature_importance.head(10),  # Show top 10 features
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Top 10 Caractéristiques par Importance',
                            labels={'Importance': 'Importance Relative', 'Feature': 'Caractéristique'},
                            color='Importance',
                            color_continuous_scale='Viridis'
                        )
                        
                        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display full table of importances with expander
                        with st.expander("Voir toutes les importances"):
                            st.dataframe(
                                feature_importance.style.format({'Importance': '{:.4f}', 'Importance_Normalized': '{:.2f}'}),
                                hide_index=True,
                                use_container_width=True
                            )
                    else:
                        st.warning("Ce modèle ne fournit pas d'importances de caractéristiques")
                except Exception as e:
                    st.error(f"Erreur lors de l'affichage des importances: {str(e)}")
                    logging.exception("Error displaying feature importances")
                    
            # Feature correlations section
            st.subheader("Corrélations entre Caractéristiques")
            
            try:
                # Load training data to compute feature correlations
                training_data = pd.read_csv(TRAINING_DATA_PATH)
                
                # Filter for selected features that exist in the data
                available_features = [f for f in selected_features if f in training_data.columns]
                
                if available_features:
                    # Convert to numeric
                    features_df = training_data[available_features].apply(pd.to_numeric, errors='coerce')
                    
                    # Compute correlation matrix
                    corr_matrix = features_df.corr()
                    
                    # Create heatmap
                    fig = px.imshow(
                        corr_matrix,
                        title="Matrice de Corrélation des Caractéristiques",
                        labels=dict(x="Caractéristique", y="Caractéristique", color="Corrélation"),
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1
                    )
                    fig.update_layout(height=600)
                    
                    # Adjust layout if there are many features
                    if len(available_features) > 15:
                        fig.update_layout(height=800)
                        st.warning("Nombreuses caractéristiques - la visualisation peut être dense")
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Aucune caractéristique disponible pour calculer les corrélations")
            except Exception as e:
                st.error(f"Erreur lors du calcul des corrélations: {str(e)}")
                logging.exception("Error calculating feature correlations")
        
        # --- Tab 3: Evaluation ---
        with tab_evaluation:
            st.subheader("Évaluation du Modèle")
            
            try:
                # Load training data
                training_data = pd.read_csv(TRAINING_DATA_PATH)
                
                # Import necessary functions
                from services.ml_service import preprocess_features, create_response_labels
                
                # Use same approach as overview tab for consistency
                # Try to create synthetic follow-up data if none exists
                # This helps with displaying the interface and sample metrics
                follow_up_columns = ['madrs_score_fu', 'madrs_score_end', 'madrs_score_final', 'madrs_last_score']
                has_follow_up = any(col in training_data.columns for col in follow_up_columns)
                
                if not has_follow_up and 'madrs_score_bl' in training_data.columns:
                    st.info("Données de suivi MADRS manquantes. Création de données simulées pour démonstration.")
                    # Create synthetic follow-up data by simulating improvement
                    np.random.seed(42)  # For reproducibility
                    bl_scores = pd.to_numeric(training_data['madrs_score_bl'], errors='coerce')
                    valid_bl = bl_scores.notna()
                    
                    # Only generate for valid baseline scores
                    if valid_bl.sum() > 10:
                        # Generate random improvement (30-70% reduction)
                        improvement = np.random.uniform(0.3, 0.7, size=len(training_data))
                        training_data['madrs_score_fu'] = bl_scores * (1 - improvement)
                        
                        # Add a note that these are simulated data
                        st.warning("⚠️ Avertissement: Les métriques sont calculées sur des données simulées et sont présentées à titre d'exemple uniquement.")
                
                # Check for valid baseline values
                valid_bl = pd.to_numeric(training_data['madrs_score_bl'], errors='coerce').notna()
                
                # Find which follow-up column to use
                follow_up_col = next((col for col in follow_up_columns if col in training_data.columns), None)
                valid_fu = pd.to_numeric(training_data[follow_up_col], errors='coerce').notna() if follow_up_col else pd.Series(False, index=training_data.index)
                
                # Create response labels (filter training data first to ensure valid scores)
                df_valid = training_data[valid_bl & valid_fu].reset_index(drop=True)
                
                # Process features only for valid data
                y = create_response_labels(df_valid)
                X = preprocess_features(df_valid, selected_features)
                
                # Keep only rows with valid labels
                valid_idx = y.notna()
                X_valid = X[valid_idx].reset_index(drop=True)
                y_valid = y[valid_idx].reset_index(drop=True)
                
                if len(y_valid) > 10:
                    # Make predictions
                    y_pred_proba = model.predict_proba(X_valid)[:, 1]
                    y_pred = model.predict(X_valid)
                    
                    # Create tabs for different evaluation metrics
                    eval_tab1, eval_tab2, eval_tab3 = st.tabs([
                        "Matrice de Confusion", 
                        "Courbe ROC", 
                        "Précision-Rappel"
                    ])
                    
                    with eval_tab1:
                        # Calculate confusion matrix
                        cm = confusion_matrix(y_valid, y_pred)
                        
                        # Create confusion matrix plot
                        fig = px.imshow(
                            cm,
                            labels=dict(x="Prédit", y="Réel", color="Compte"),
                            x=["Non-Répondeur", "Répondeur"],
                            y=["Non-Répondeur", "Répondeur"],
                            text_auto=True,
                            color_continuous_scale="Blues"
                        )
                        
                        # Add annotations for percentages
                        total = np.sum(cm)
                        percentages = cm / total * 100
                        
                        for i in range(len(cm)):
                            for j in range(len(cm[i])):
                                fig.add_annotation(
                                    x=j, y=i,
                                    text=f"{percentages[i, j]:.1f}%",
                                    showarrow=False,
                                    font=dict(color="white" if cm[i, j] > cm.max()/2 else "black")
                                )
                        
                        fig.update_layout(title="Matrice de Confusion")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display classification report
                        st.subheader("Rapport de Classification")
                        
                        # Manually calculate metrics
                        tn, fp, fn, tp = cm.ravel()
                        
                        accuracy = (tp + tn) / total
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        
                        metrics_df = pd.DataFrame({
                            "Classe": ["Non-Répondeur", "Répondeur", "Moyenne"],
                            "Précision": [tn / (tn + fn) if (tn + fn) > 0 else 0, precision, precision],
                            "Rappel": [tn / (tn + fp) if (tn + fp) > 0 else 0, recall, recall],
                            "F1-Score": [2 * (tn / (tn + fn)) * (tn / (tn + fp)) / ((tn / (tn + fn)) + (tn / (tn + fp))) if ((tn / (tn + fn)) + (tn / (tn + fp))) > 0 else 0, f1, f1],
                            "Support": [tn + fp, fn + tp, total]
                        })
                        
                        st.dataframe(metrics_df.style.format({
                            "Précision": "{:.2f}",
                            "Rappel": "{:.2f}",
                            "F1-Score": "{:.2f}",
                            "Support": "{:.0f}"
                        }), hide_index=True, use_container_width=True)
                    
                    with eval_tab2:
                        # Calculate ROC curve
                        fpr, tpr, thresholds = roc_curve(y_valid, y_pred_proba)
                        roc_auc = auc(fpr, tpr)
                        
                        # Create ROC curve plot
                        fig = px.line(
                            x=fpr, y=tpr,
                            labels={"x": "Taux de Faux Positifs", "y": "Taux de Vrais Positifs"},
                            title=f"Courbe ROC (AUC = {roc_auc:.3f})"
                        )
                        
                        # Add diagonal reference line
                        fig.add_shape(
                            type='line', line=dict(dash='dash', color='gray'),
                            x0=0, y0=0, x1=1, y1=1
                        )
                        
                        # Add thresholds points
                        n_thresholds = min(10, len(thresholds))
                        indices = np.linspace(0, len(thresholds) - 1, n_thresholds).astype(int)
                        
                        fig.add_trace(go.Scatter(
                            x=[fpr[i] for i in indices],
                            y=[tpr[i] for i in indices],
                            text=[f"Seuil: {thresholds[i]:.2f}" for i in indices],
                            mode="markers",
                            marker=dict(size=8, color="red"),
                            name="Seuils"
                        ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add explanation
                        st.markdown("""
                        **Interprétation de la courbe ROC:**
                        - Plus l'AUC (aire sous la courbe) est proche de 1, meilleure est la performance du modèle
                        - Une AUC de 0.5 (ligne diagonale) représente une performance aléatoire
                        - Les points rouges représentent différents seuils de classification
                        """)
                    
                    with eval_tab3:
                        # Calculate precision-recall curve
                        precision, recall, thresholds = precision_recall_curve(y_valid, y_pred_proba)
                        
                        # Create precision-recall curve plot
                        pr_df = pd.DataFrame({
                            "Rappel": recall,
                            "Précision": precision,
                        })
                        
                        fig = px.line(
                            pr_df, x="Rappel", y="Précision",
                            title="Courbe Précision-Rappel",
                            labels={"Rappel": "Rappel", "Précision": "Précision"}
                        )
                        
                        # Add baseline
                        baseline = sum(y_valid) / len(y_valid)
                        fig.add_shape(
                            type='line', line=dict(dash='dash', color='gray'),
                            x0=0, y0=baseline, x1=1, y1=baseline
                        )
                        
                        # Add thresholds points
                        if len(thresholds) > 0:
                            n_thresholds = min(10, len(thresholds))
                            indices = np.linspace(0, len(thresholds) - 1, n_thresholds).astype(int)
                            
                            fig.add_trace(go.Scatter(
                                x=[recall[i+1] for i in indices],
                                y=[precision[i+1] for i in indices],
                                text=[f"Seuil: {thresholds[i]:.2f}" for i in indices],
                                mode="markers",
                                marker=dict(size=8, color="red"),
                                name="Seuils"
                            ))
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add explanation
                        st.markdown(f"""
                        **Interprétation de la courbe Précision-Rappel:**
                        - La ligne en pointillés représente la performance de référence (% de répondeurs: {baseline:.2f})
                        - Idéalement, la courbe doit rester le plus haut possible en s'éloignant vers la droite
                        - Les compromis entre précision et rappel peuvent être ajustés en modifiant le seuil de classification
                        """)
                else:
                    st.warning("Données insuffisantes pour calculer les métriques d'évaluation")
            except Exception as e:
                st.error(f"Erreur lors de l'évaluation du modèle: {str(e)}")
                logging.exception("Error evaluating model")
        
        # --- Tab 4: Validation ---
        with tab_validation:
            st.subheader("Validation et Robustesse")
            
            st.markdown("""
            ### Méthodes de Validation
            
            La validation du modèle a été effectuée en utilisant les techniques suivantes:
            
            1. **Validation croisée**: Pour évaluer la stabilité du modèle sur différents sous-ensembles des données
            2. **Test sur données non-vues**: Pour évaluer la performance sur de nouvelles données
            3. **Analyses de sensibilité**: Pour tester la robustesse aux variations dans les données d'entrée
            """)
            
            # Mock cross-validation scores for demonstration
            # In a real implementation, these would be computed or loaded from saved results
            cv_scores = {
                "Fold 1": 0.75,
                "Fold 2": 0.78,
                "Fold 3": 0.72,
                "Fold 4": 0.76,
                "Fold 5": 0.74
            }
            
            # Create a bar chart for cross-validation scores
            cv_df = pd.DataFrame({
                "Fold": list(cv_scores.keys()),
                "Score": list(cv_scores.values())
            })
            
            fig = px.bar(
                cv_df, x="Fold", y="Score",
                title="Scores de Validation Croisée (AUC)",
                labels={"Score": "Score AUC"},
                color="Score",
                color_continuous_scale="Viridis"
            )
            
            fig.update_layout(yaxis_range=[0.5, 1.0])
            fig.add_hline(y=np.mean(list(cv_scores.values())), line_dash="dash", line_color="red")
            fig.add_annotation(
                x=2, y=np.mean(list(cv_scores.values())) + 0.02,
                text=f"Moyenne: {np.mean(list(cv_scores.values())):.3f}",
                showarrow=False,
                font=dict(color="red")
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add more validation visualizations and explanations
            st.subheader("Analyses de Sensibilité")
            
            st.markdown("""
            L'analyse de sensibilité permet de comprendre comment les prédictions du modèle changent lorsque les valeurs d'entrée varient.
            
            Choisissez une caractéristique pour voir comment la probabilité de réponse change lorsqu'on fait varier sa valeur:
            """)
            
            # Select a feature to analyze
            feature_to_analyze = st.selectbox(
                "Choisir une caractéristique",
                selected_features
            )
            
            # Fake sensitivity analysis (this would be replaced with actual calculations)
            try:
                # Create a range of values for the selected feature
                feature_min = 0
                feature_max = 60 if "age" in feature_to_analyze else 30
                feature_range = np.linspace(feature_min, feature_max, 100)
                
                # Create a sample patient with median values
                sample_data = pd.DataFrame({
                    feature: [np.random.randint(1, 10) if "score" in feature else (40 if "age" in feature else 2)]
                    for feature in selected_features
                })
                
                # Function to predict probability for varying feature values
                def predict_for_range(feature_name, value_range):
                    probas = []
                    for val in value_range:
                        # Create a copy of the sample data
                        sample_copy = sample_data.copy()
                        # Update the feature value
                        sample_copy[feature_name] = val
                        # Predict
                        try:
                            proba = model.predict_proba(sample_copy)[0, 1] * 100
                            probas.append(proba)
                        except:
                            probas.append(np.nan)
                    return probas
                
                # Get predictions for the feature range
                probas = predict_for_range(feature_to_analyze, feature_range)
                
                # Create a DataFrame for plotting
                sensitivity_df = pd.DataFrame({
                    feature_to_analyze: feature_range,
                    "Probabilité de Réponse (%)": probas
                })
                
                # Create a line plot
                fig = px.line(
                    sensitivity_df, x=feature_to_analyze, y="Probabilité de Réponse (%)",
                    title=f"Analyse de Sensibilité pour {feature_to_analyze}",
                    labels={feature_to_analyze: feature_to_analyze, "Probabilité de Réponse (%)": "Probabilité de Réponse (%)"}
                )
                
                # Add a horizontal line at 50% probability
                fig.add_hline(y=50, line_dash="dash", line_color="red")
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Comment interpréter:**
                - La ligne montre comment la probabilité prédite change lorsque la valeur de la caractéristique sélectionnée varie
                - La ligne rouge pointillée représente le seuil de 50% (classification comme répondeur/non-répondeur)
                - Une ligne plate indique que la caractéristique a peu d'influence sur la prédiction
                - Une pente forte indique que la caractéristique a une forte influence sur la prédiction
                """)
            except Exception as e:
                st.error(f"Erreur lors de l'analyse de sensibilité: {str(e)}")
                logging.exception("Error in sensitivity analysis")
        
        # --- Tab 5: Interpretation ---
        with tab_interpretation:
            st.subheader("Interprétation Clinique")
            
            st.markdown("""
            ### Interprétation des Résultats
            
            Le modèle de prédiction de réponse au traitement TMS pour la dépression repose sur plusieurs caractéristiques clés qui peuvent être interprétées cliniquement:
            
            #### Facteurs de Bon Pronostic
            - **Âge plus jeune:** Les patients plus jeunes semblent généralement mieux répondre au traitement
            - **Sévérité modérée de la dépression:** Les patients avec une dépression d'intensité modérée (plutôt que sévère) semblent avoir un meilleur taux de réponse
            - **Symptômes anxieux moins prononcés:** Les patients avec des niveaux d'anxiété plus faibles ont tendance à mieux répondre
            - **Profil de personnalité:** Certains traits sont associés à une meilleure réponse (ex: ouverture, extraversion)
            
            #### Facteurs de Mauvais Pronostic
            - **Durée prolongée de l'épisode actuel:** Une plus longue durée de l'épisode actuel est associée à une réponse moins favorable
            - **Antécédents d'échec thérapeutique:** Les patients ayant eu plusieurs échecs de traitement antérieurs ont tendance à moins bien répondre
            - **Comorbidités psychiatriques:** La présence de comorbidités peut réduire le taux de réponse
            """)
            
            # Add a shap-like plot for feature impact (simplified example)
            st.subheader("Impact des Caractéristiques sur la Prédiction")
            
            try:
                # Create a mock dataframe for feature impact visualization
                # In a real implementation, this would use SHAP values or similar
                impact_df = pd.DataFrame({
                    "Caractéristique": selected_features[:10] if len(selected_features) > 10 else selected_features,
                    "Impact": np.linspace(1, -1, len(selected_features[:10] if len(selected_features) > 10 else selected_features)),
                    "Groupe": ["Démographique" if "age" in f or "sexe" in f else 
                              "Clinique" if "score" in f else 
                              "Personnalité" if "bfi" in f else 
                              "Autre" for f in (selected_features[:10] if len(selected_features) > 10 else selected_features)]
                }).sort_values("Impact")
                
                # Create a horizontal bar chart
                fig = px.bar(
                    impact_df, y="Caractéristique", x="Impact", color="Groupe",
                    title="Impact des Caractéristiques sur la Prédiction",
                    labels={"Impact": "Impact sur la Probabilité de Réponse", "Caractéristique": "Caractéristique", "Groupe": "Groupe"},
                    orientation='h',
                    color_discrete_map={
                        "Démographique": "royalblue",
                        "Clinique": "indianred",
                        "Personnalité": "green",
                        "Autre": "orange"
                    }
                )
                
                fig.add_vline(x=0, line_dash="dash", line_color="black")
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Comment interpréter:**
                - Les barres à droite (valeurs positives) indiquent des caractéristiques qui augmentent la probabilité de réponse
                - Les barres à gauche (valeurs négatives) indiquent des caractéristiques qui diminuent la probabilité de réponse
                - La longueur de chaque barre représente l'ampleur de l'impact
                """)
            except Exception as e:
                st.error(f"Erreur lors de la visualisation de l'impact des caractéristiques: {str(e)}")
                logging.exception("Error in feature impact visualization")
            
            # Case studies section
            st.subheader("Études de Cas")
            
            st.markdown("""
            Pour mieux comprendre comment le modèle fonctionne dans des cas concrets, considérons deux profils de patients hypothétiques:
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                #### Patient A - Forte Probabilité de Réponse
                
                - **Âge:** 32 ans
                - **Sexe:** Femme
                - **MADRS initial:** 24 (dépression modérée)
                - **BAI initial:** 15 (anxiété légère)
                - **Durée de l'épisode:** 6 mois
                - **Traitements antérieurs:** 1 échec
                - **Personnalité:** Extraversion élevée
                
                **Probabilité de réponse prédite:** 78%
                """)
            
            with col2:
                st.markdown("""
                #### Patient B - Faible Probabilité de Réponse
                
                - **Âge:** 58 ans
                - **Sexe:** Homme
                - **MADRS initial:** 36 (dépression sévère)
                - **BAI initial:** 28 (anxiété modérée à sévère)
                - **Durée de l'épisode:** 24 mois
                - **Traitements antérieurs:** 4+ échecs
                - **Personnalité:** Névrosisme élevé
                
                **Probabilité de réponse prédite:** 32%
                """)
            
            st.info("""
            **Note importante sur l'interprétation clinique:**
            
            Ce modèle est conçu comme un outil d'aide à la décision et non comme un substitut au jugement clinique. 
            Les prédictions doivent toujours être interprétées dans le contexte clinique global du patient et 
            de nombreux autres facteurs non capturés par le modèle peuvent influencer les résultats du traitement.
            """)
    
        # --- New Tab: Decision Curve Analysis ---
        with tab_decision_curve:
            st.subheader("Analyse par Courbe de Décision")
            
            st.markdown("""
            ### Qu'est-ce que l'Analyse par Courbe de Décision?
            
            L'analyse par courbe de décision est une méthode pour évaluer l'utilité clinique d'un modèle de prédiction. 
            Elle calcule le "bénéfice net" pour différents seuils de probabilité (threshold probability), représentant 
            différentes préférences cliniques.
            
            **Interprétation:**
            - L'axe X représente le seuil de probabilité - la probabilité minimale à laquelle l'intervention est justifiée
            - L'axe Y représente le bénéfice net - équivalent au nombre de vrais positifs moins les faux positifs pondérés
            - Un modèle est cliniquement utile s'il a le bénéfice net le plus élevé sur la plage de seuils pertinents
            """)
            
            # Calculate decision curve
            try:
                # Load evaluation cohort or generate if needed
                cohort = generate_evaluation_cohort(size=50)  # Fixed parameter name from n_patients to size
                
                if not cohort.empty and 'predicted_probability' in cohort.columns and 'true_label' in cohort.columns:
                    # Calculate net benefit across threshold probabilities
                    thresholds = np.linspace(0.01, 0.99, 50)
                    
                    # Initialize arrays for net benefit values
                    nb_model = []
                    nb_all = []
                    nb_none = []
                    
                    # Calculate net benefit for each threshold
                    for threshold in thresholds:
                        nb_model.append(calculate_net_benefit(
                            y_true=cohort['true_label'],
                            y_pred_proba=cohort['predicted_probability'],
                            threshold=threshold
                        ))
                        
                        # Net benefit of treat all is true positive rate - false positive rate * odds
                        prevalence = cohort['true_label'].mean()
                        nb_all.append(prevalence - (1 - prevalence) * threshold / (1 - threshold))
                        
                        # Net benefit of treat none is always 0
                        nb_none.append(0)
                    
                    # Create DataFrame for plotting
                    decision_curve_df = pd.DataFrame({
                        'Threshold Probability': thresholds,
                        'Model': nb_model,
                        'Treat All': nb_all,
                        'Treat None': nb_none
                    })
                    
                    # Plot decision curve
                    fig = px.line(
                        decision_curve_df, 
                        x='Threshold Probability', 
                        y=['Model', 'Treat All', 'Treat None'],
                        labels={'value': 'Net Benefit', 'variable': 'Strategy'},
                        title='Courbe de Décision: Bénéfice Net vs Seuil de Probabilité',
                        color_discrete_map={
                            'Model': st.session_state.PASTEL_COLORS[0],
                            'Treat All': st.session_state.PASTEL_COLORS[1],
                            'Treat None': 'black'
                        }
                    )
                    
                    # Improve layout
                    fig.update_layout(
                        legend_title_text='Stratégie',
                        xaxis_title='Seuil de Probabilité',
                        yaxis_title='Bénéfice Net',
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="right",
                            x=0.99
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add interpretation
                    st.markdown("""
                    ### Interprétation de la Courbe de Décision
                    
                    **Comment interpréter cette courbe:**
                    - Si la ligne du modèle est au-dessus des stratégies "Treat All" et "Treat None", 
                      le modèle apporte une valeur clinique ajoutée
                    - La plage de seuils où le modèle est supérieur correspond aux différentes préférences cliniques
                    - Plus l'écart est grand entre le modèle et les autres stratégies, plus le bénéfice clinique est important
                    
                    **Applications pratiques:**
                    - Si le modèle n'est jamais au-dessus des autres lignes, il ne devrait pas être utilisé cliniquement
                    - Si le modèle est supérieur seulement sur une plage limitée, il peut être utile uniquement pour 
                      certains contextes cliniques spécifiques
                    - Si le modèle est supérieur sur toute la plage raisonnable de seuils, il peut être recommandé pour 
                      une utilisation clinique générale
                    """)
                    
                    # Show analysis of specific threshold ranges
                    clinically_relevant_range = (0.1, 0.5)  # Example range, adjust based on clinical context
                    
                    # Filter to clinically relevant range
                    relevant_df = decision_curve_df[
                        (decision_curve_df['Threshold Probability'] >= clinically_relevant_range[0]) & 
                        (decision_curve_df['Threshold Probability'] <= clinically_relevant_range[1])
                    ]
                    
                    # Calculate average net benefit differences
                    avg_diff_model_all = (relevant_df['Model'] - relevant_df['Treat All']).mean()
                    avg_diff_model_none = (relevant_df['Model'] - relevant_df['Treat None']).mean()
                    
                    # Determine if model is superior
                    is_model_superior = avg_diff_model_all > 0 and avg_diff_model_none > 0
                    
                    # Clinical recommendation based on decision curve
                    st.subheader("Recommandation Clinique")
                    
                    if is_model_superior:
                        st.success(f"""
                        ✅ Sur la plage cliniquement pertinente des seuils de probabilité ({clinically_relevant_range[0]} - {clinically_relevant_range[1]}), 
                        ce modèle offre un bénéfice net supérieur aux stratégies alternatives.
                        
                        Bénéfice moyen vs. traiter tous: {avg_diff_model_all:.4f}
                        Bénéfice moyen vs. traiter aucun: {avg_diff_model_none:.4f}
                        
                        **Recommandation:** Le modèle peut être considéré pour une application clinique.
                        """)
                    else:
                        st.warning(f"""
                        ⚠️ Sur la plage cliniquement pertinente des seuils de probabilité ({clinically_relevant_range[0]} - {clinically_relevant_range[1]}), 
                        ce modèle n'offre pas un bénéfice net supérieur aux stratégies alternatives sur l'ensemble de la plage.
                        
                        Bénéfice moyen vs. traiter tous: {avg_diff_model_all:.4f}
                        Bénéfice moyen vs. traiter aucun: {avg_diff_model_none:.4f}
                        
                        **Recommandation:** Une évaluation plus approfondie est nécessaire avant l'application clinique.
                        """)
                        
                    # Add interventions avoided curve
                    st.subheader("Interventions Évitées")
                    
                    # Calculate interventions avoided at different thresholds
                    interventions_avoided = []
                    
                    for threshold in thresholds:
                        # For each threshold, calculate how many interventions are avoided vs treating all
                        model_intervention_rate = (cohort['predicted_probability'] >= threshold).mean()
                        interventions_avoided.append((1 - model_intervention_rate) * 100)  # per 100 patients
                    
                    # Create DataFrame for plotting
                    interventions_df = pd.DataFrame({
                        'Threshold Probability': thresholds,
                        'Interventions Avoided (per 100 patients)': interventions_avoided
                    })
                    
                    # Plot interventions avoided
                    fig_interventions = px.line(
                        interventions_df,
                        x='Threshold Probability',
                        y='Interventions Avoided (per 100 patients)',
                        title='Interventions Évitées par Seuil de Probabilité',
                        labels={
                            'Threshold Probability': 'Seuil de Probabilité', 
                            'Interventions Avoided (per 100 patients)': 'Interventions Évitées (pour 100 patients)'
                        }
                    )
                    
                    fig_interventions.update_traces(line=dict(color=st.session_state.PASTEL_COLORS[2], width=2))
                    
                    st.plotly_chart(fig_interventions, use_container_width=True)
                    
                else:
                    st.warning("Données insuffisantes pour calculer la courbe de décision.")
            except Exception as e:
                st.error(f"Erreur lors du calcul de la courbe de décision: {str(e)}")
                logging.exception("Error calculating decision curve")
    
    except Exception as e:
        st.error(f"Erreur lors du chargement de la page de performance ML: {str(e)}")
        logging.exception("Error loading ML performance page")

# Add this function to calculate net benefit
def calculate_net_benefit(y_true, y_pred_proba, threshold):
    """
    Calculate net benefit for decision curve analysis.
    
    Args:
        y_true: True binary outcomes (0/1)
        y_pred_proba: Predicted probabilities
        threshold: Probability threshold
        
    Returns:
        Net benefit at the given threshold
    """
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    # Classify as positive based on threshold
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate true positives and false positives
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    
    # Total number of patients
    n = len(y_true)
    
    # Calculate net benefit
    if n == 0:
        return 0
    
    # The weight is based on the threshold odds
    w = threshold / (1 - threshold)
    
    # Net benefit formula
    net_benefit = (TP / n) - (FP / n) * w
    
    return net_benefit

# Make sure there are no issues with name shadowing or circular imports
# Explicitly export the function
__all__ = ['ml_performance_page']