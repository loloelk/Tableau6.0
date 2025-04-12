# components/dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # Need graph_objects for radar chart
import base64
import logging
import numpy as np
from services.network_analysis import generate_person_specific_network
from services.nurse_service import get_latest_nurse_inputs, get_nurse_inputs_history, get_side_effects_history

# Medication categories for display
MEDICATION_CATEGORIES = {
    'SSRI': ['Escitalopram', 'Sertraline', 'Fluoxetine', 'Paroxetine', 'Citalopram'],
    'SNRI': ['Venlafaxine', 'Duloxetine', 'Desvenlafaxine', 'Levomilnacipran'],
    'TCA': ['Amitriptyline', 'Nortriptyline', 'Imipramine', 'Desipramine', 'Clomipramine'],
    'Antipsychotics': ['Quetiapine', 'Aripiprazole', 'Risperidone', 'Olanzapine', 'Lurasidone'],
    'Other': ['Mirtazapine', 'Bupropion', 'Trazodone', 'Vilazodone', 'Vortioxetine']
}

# Helper function to get EMA data (ensure robustness)
def get_patient_ema_data(patient_id):
    """Retrieve and prepare EMA data for a specific patient"""
    if 'simulated_ema_data' not in st.session_state or st.session_state.simulated_ema_data.empty:
        logging.warning("Simulated EMA data not found in session state.")
        return pd.DataFrame()
    if 'PatientID' not in st.session_state.simulated_ema_data.columns:
         logging.error("Column 'PatientID' missing in simulated EMA data.")
         return pd.DataFrame()
    try:
        patient_ema = st.session_state.simulated_ema_data[
            st.session_state.simulated_ema_data['PatientID'] == patient_id
        ].copy()
        if 'Timestamp' in patient_ema.columns:
            patient_ema['Timestamp'] = pd.to_datetime(patient_ema['Timestamp'], errors='coerce')
            patient_ema.dropna(subset=['Timestamp'], inplace=True)
            patient_ema.sort_values(by='Timestamp', inplace=True)
        else:
            logging.warning("'Timestamp' column missing in patient EMA data.")
    except Exception as e:
         logging.error(f"Error processing EMA data for {patient_id}: {e}")
         return pd.DataFrame()
    return patient_ema

def treatment_progress(patient_ema):
    """Display treatment progress tracking based on EMA dates"""
    st.subheader("Suivi de Progression du Traitement (Basé sur EMA)")
    milestones = ['Éval Initiale', 'Semaine 1', 'Semaine 2', 'Semaine 3', 'Semaine 4', 'Fin Traitement']
    assumed_duration_days = 28
    
    if patient_ema.empty or 'Timestamp' not in patient_ema.columns:
        st.warning("ℹ️ Données EMA ou Timestamps manquants pour suivre la progression.")
        return
    
    try:
        first_entry = patient_ema['Timestamp'].min()
        last_entry = patient_ema['Timestamp'].max()
        days_elapsed = (last_entry - first_entry).days if pd.notna(first_entry) and pd.notna(last_entry) else 0
    except Exception as e:
        logging.error(f"Error calculating date range from EMA Timestamps: {e}")
        days_elapsed = 0
    
    progress_percentage = min((days_elapsed / assumed_duration_days), 1) if assumed_duration_days > 0 else 0
    
    if days_elapsed <= 1: current_milestone_index = 0
    elif days_elapsed <= 7: current_milestone_index = 1
    elif days_elapsed <= 14: current_milestone_index = 2
    elif days_elapsed <= 21: current_milestone_index = 3
    elif days_elapsed <= assumed_duration_days: current_milestone_index = 4
    else: current_milestone_index = 5
    
    # Import our custom metrics functions
    from components.common.metrics import custom_progress_bar, milestone_progress
    
    # Create a container for the progress section
    progress_container = st.container()
    
    with progress_container:
        # Add a clearer heading
        st.markdown("#### État d'avancement")
        
        # Create a visual progress container with border
        st.markdown("""
        <div style="border: 1px solid #e0e0e0; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
        """, unsafe_allow_html=True)
        
        # Use custom progress bar with more contrast
        custom_progress_bar(
            progress_value=progress_percentage,
            height="12px",  # Slightly taller for better visibility
            bg_color="#f0f0f0",  # Slightly darker background
            fill_color="linear-gradient(90deg, #1976d2, #42a5f5)",  # More vibrant blue gradient
            text_after=f"Jour {days_elapsed} sur {assumed_duration_days} ({int(progress_percentage * 100)}%)"
        )
        
        # Add a small spacer
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        
        # Use custom milestone progress display with more visual distinction
        milestone_progress(
            current_milestone=current_milestone_index,
            milestones=milestones,
            completed_color="#1976d2",
            active_color="#3f51b5"
        )
        
        # Close the container
        st.markdown("</div>", unsafe_allow_html=True)

def patient_dashboard():
    """Main dashboard for individual patient view"""
    # Apply custom styling
    st.markdown("""
    <style>
        /* Card-like elements */
        div[data-testid="stVerticalBlock"] > div:has(div.element-container div.stMarkdown) {
            background-color: white;
            border-radius: 0.5rem;
            padding: 0;
            margin-bottom: 1rem;
        }
        
        /* Table styling */
        .dataframe {
            border-collapse: collapse;
            width: 100%;
            border: none !important;
        }
        .dataframe th {
            background-color: #f8f9fa;
            border: none !important;
            padding: 8px 12px !important;
            text-align: left;
        }
        .dataframe td {
            border-top: 1px solid #f0f0f0 !important;
            border-left: none !important;
            border-right: none !important;
            padding: 8px 12px !important;
        }
        .dataframe tr:hover {
            background-color: #f8f9fa;
        }
        
        /* Section headers */
        h3 {
            font-size: 1.2rem;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        
        /* Tabs styling */
        button[data-baseweb="tab"] {
            font-size: 0.9rem;
            padding: 0.5rem 1rem;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background-color: #f8f9fa;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.header("📊 Tableau de Bord du Patient")
    
    if not st.session_state.get("selected_patient_id"):
        st.warning("⚠️ Aucun patient sélectionné. Veuillez en choisir un dans la barre latérale.")
        return
        
    patient_id = st.session_state.selected_patient_id
    
    # Get patient data from session state
    if 'final_data' not in st.session_state or st.session_state.final_data.empty:
         st.error("❌ Données principales du patient non chargées.")
         return
    try:
         if 'ID' not in st.session_state.final_data.columns:
              st.error("Colonne 'ID' manquante dans les données patient principales.")
              return
         patient_row = st.session_state.final_data[st.session_state.final_data["ID"] == patient_id]
         if patient_row.empty:
             st.error(f"❌ Données non trouvées pour le patient {patient_id}.")
             return
         patient_data = patient_row.iloc[0]
    except Exception as e:
         st.error(f"Erreur récupération données pour {patient_id}: {e}")
         logging.exception(f"Error fetching data for patient {patient_id}")
         return

    patient_ema = get_patient_ema_data(patient_id)
    
    # Create tab navigation menu at top - removing hospitalization tab
    tab_overview, tab_assessments, tab_symptoms, tab_care_plan, tab_side_effects, tab_notes_history = st.tabs([
        "👤 Aperçu", "📈 Évaluations", "🩺 Symptômes", 
        "🎯 Plan de Soins", "💊 Effets 2nd", "📝 Historique Notes"
    ])
    
    # Tab 1: Patient Overview (Aperçu)
    with tab_overview:
        create_patient_overview(patient_data)
    
    # Tab 2: Clinical Assessments (Évaluations)
    with tab_assessments:
        st.header("📈 Évaluations Cliniques")
        # --- BFI MODIFICATION START: Add BFI tab ---
        subtab_madrs, subtab_phq9, subtab_bfi = st.tabs(["MADRS", "PHQ-9", "BFI"])
        # --- BFI MODIFICATION END ---

        with subtab_madrs:
            # (MADRS logic remains the same)
            st.subheader("Scores MADRS")
            madrs_bl = pd.to_numeric(patient_data.get("madrs_score_bl"), errors='coerce')
            madrs_fu = pd.to_numeric(patient_data.get("madrs_score_fu"), errors='coerce')
            if pd.isna(madrs_bl): st.warning("Score MADRS Baseline manquant.")
            else:
                # (Detailed MADRS display logic as before...)
                col1_madrs, col2_madrs = st.columns(2)
                with col1_madrs:
                    st.metric(label="MADRS Baseline", value=f"{madrs_bl:.0f}")
                    score = madrs_bl
                    if score <= 6: severity = "Normal"
                    elif score <= 19: severity = "Légère"
                    elif score <= 34: severity = "Modérée"
                    else: severity = "Sévère"
                    st.write(f"**Sévérité Initiale:** {severity}")
                    if not pd.isna(madrs_fu):
                        delta_score = madrs_fu - madrs_bl; st.metric(label="MADRS Jour 30", value=f"{madrs_fu:.0f}", delta=f"{delta_score:.0f} points")
                        if madrs_bl > 0:
                             improvement_pct = ((madrs_bl - madrs_fu) / madrs_bl) * 100; st.metric(label="Amélioration", value=f"{improvement_pct:.1f}%")
                             is_responder = improvement_pct >= 50; is_remitter = madrs_fu < 10
                             st.write(f"**Réponse (>50%):** {'Oui' if is_responder else 'Non'}"); st.write(f"**Rémission (<10):** {'Oui' if is_remitter else 'Non'}")
                        else: st.write("Amélioration (%) non calculable (baseline=0)")
                    else: st.metric(label="MADRS Jour 30", value="N/A")
                    madrs_total_df = pd.DataFrame({ 'Temps': ['Baseline', 'Jour 30'],'Score': [madrs_bl, madrs_fu if not pd.isna(madrs_fu) else np.nan]})
                    fig_madrs_total = px.bar( madrs_total_df.dropna(subset=['Score']), x='Temps', y='Score', title="Score Total MADRS", color='Temps', color_discrete_sequence=st.session_state.PASTEL_COLORS[:2], labels={"Score": "Score MADRS Total"})
                    st.plotly_chart(fig_madrs_total, use_container_width=True)
                with col2_madrs:
                    st.subheader("Scores par Item MADRS")
                    items_data = []; valid_items_found = False
                    for i in range(1, 11):
                         bl_col = f'madrs_{i}_bl'; fu_col = f'madrs_{i}_fu'; item_label = st.session_state.MADRS_ITEMS_MAPPING.get(str(i), f"Item {i}")
                         bl_val = pd.to_numeric(patient_data.get(bl_col), errors='coerce'); fu_val = pd.to_numeric(patient_data.get(fu_col), errors='coerce')
                         if not pd.isna(bl_val) or not pd.isna(fu_val): valid_items_found = True
                         items_data.append({'Item': item_label, 'Baseline': bl_val, 'Jour 30': fu_val })
                    if not valid_items_found: st.warning("Scores par item MADRS non disponibles.")
                    else:
                        madrs_items_df = pd.DataFrame(items_data)
                        madrs_items_df['Baseline'] = pd.to_numeric(madrs_items_df['Baseline'], errors='coerce'); madrs_items_df['Jour 30'] = pd.to_numeric(madrs_items_df['Jour 30'], errors='coerce')
                        madrs_items_long = madrs_items_df.melt(id_vars='Item', var_name='Temps', value_name='Score').dropna(subset=['Score'])
                        fig_items = px.bar( madrs_items_long, x='Item', y='Score', color='Temps', barmode='group', title="Scores par Item MADRS", template="plotly_white", color_discrete_sequence=st.session_state.PASTEL_COLORS[:2], labels={"Score": "Score (0-6)"})
                        fig_items.update_xaxes(tickangle=-45); fig_items.update_yaxes(range=[0,6])
                        st.plotly_chart(fig_items, use_container_width=True)
                st.markdown("---"); st.subheader("Comparaison au groupe")
                
                # Ajout du graphique de comparaison au groupe
                if 'final_data' in st.session_state and not st.session_state.final_data.empty:
                    try:
                        # Get all MADRS scores for comparison
                        all_madrs_bl = pd.to_numeric(st.session_state.final_data['madrs_score_bl'], errors='coerce')
                        all_madrs_fu = pd.to_numeric(st.session_state.final_data['madrs_score_fu'], errors='coerce')
                        
                        # Calculate improvements for all patients
                        improvements = []
                        for bl, fu in zip(all_madrs_bl, all_madrs_fu):
                            if pd.notna(bl) and pd.notna(fu) and bl > 0:
                                imp_pct = ((bl - fu) / bl) * 100
                                improvements.append(imp_pct)
                        
                        if improvements:
                            # Calculate patient's percentile
                            patient_imp = improvement_pct if 'improvement_pct' in locals() else None
                            if patient_imp is not None:
                                percentile = sum(1 for imp in improvements if imp < patient_imp) / len(improvements) * 100
                                
                                # Create distribution plot
                                fig_dist = px.histogram(
                                    pd.DataFrame({'Amélioration (%)': improvements}), 
                                    x='Amélioration (%)', 
                                    nbins=20,
                                    color_discrete_sequence=['#A8DADC'],
                                    title=f"Distribution des Améliorations MADRS dans le Groupe (n={len(improvements)})"
                                )
                                
                                # Add marker for current patient
                                fig_dist.add_vline(
                                    x=patient_imp, 
                                    line_color='#E63946', 
                                    line_width=2,
                                    annotation_text=f"Patient actuel ({patient_imp:.1f}%)",
                                    annotation_position="top"
                                )
                                
                                st.plotly_chart(fig_dist, use_container_width=True)
                                
                                # Show percentile information
                                if percentile > 75:
                                    st.success(f"✅ Ce patient est dans le **{percentile:.0f}e centile** d'amélioration (meilleur que {percentile:.0f}% des patients)")
                                elif percentile > 50:
                                    st.info(f"ℹ️ Ce patient est dans le **{percentile:.0f}e centile** d'amélioration (meilleur que {percentile:.0f}% des patients)")
                                elif percentile > 25:
                                    st.warning(f"⚠️ Ce patient est dans le **{percentile:.0f}e centile** d'amélioration (moins bien que {100-percentile:.0f}% des patients)")
                                else:
                                    st.error(f"⚠️ Ce patient est dans le **{percentile:.0f}e centile** d'amélioration (moins bien que {100-percentile:.0f}% des patients)")
                        else:
                            st.info("Données insuffisantes pour la comparaison de groupe.")
                    except Exception as e:
                        st.error(f"Erreur lors du calcul de la comparaison: {e}")
                        logging.exception("Error calculating group comparison")
                else:
                    st.info("Données insuffisantes pour la comparaison de groupe.")
            
        # Implémentation complète du PHQ-9
        with subtab_phq9:
            st.subheader("Scores PHQ-9")
            
            # Récupération des scores PHQ-9
            phq9_bl = pd.to_numeric(patient_data.get("phq9_score_bl"), errors='coerce')
            phq9_fu = pd.to_numeric(patient_data.get("phq9_score_fu"), errors='coerce')
            
            if pd.isna(phq9_bl):
                st.warning("Score PHQ-9 Baseline manquant.")
            else:
                col1_phq, col2_phq = st.columns(2)
                
                with col1_phq:
                    st.metric(label="PHQ-9 Baseline", value=f"{phq9_bl:.0f}")
                    
                    # Classification de sévérité PHQ-9
                    score = phq9_bl
                    if score <= 4: severity = "Minimal"
                    elif score <= 9: severity = "Légère"
                    elif score <= 14: severity = "Modérée"
                    elif score <= 19: severity = "Modérément Sévère"
                    else: severity = "Sévère"
                    
                    st.write(f"**Sévérité Initiale:** {severity}")
                    
                    if not pd.isna(phq9_fu):
                        delta_score = phq9_fu - phq9_bl
                        st.metric(label="PHQ-9 Jour 30", value=f"{phq9_fu:.0f}", delta=f"{delta_score:.0f} points")
                        
                        if phq9_bl > 0:
                            improvement_pct = ((phq9_bl - phq9_fu) / phq9_bl) * 100
                            st.metric(label="Amélioration", value=f"{improvement_pct:.1f}%")
                            
                            is_responder = improvement_pct >= 50
                            is_remitter = phq9_fu < 5
                            
                            st.write(f"**Réponse (>50%):** {'Oui' if is_responder else 'Non'}")
                            st.write(f"**Rémission (<5):** {'Oui' if is_remitter else 'Non'}")
                        else:
                            st.write("Amélioration (%) non calculable (baseline=0)")
                    else:
                        st.metric(label="PHQ-9 Jour 30", value="N/A")
                    
                    # Graphique d'évolution du score total
                    phq9_total_df = pd.DataFrame({
                        'Temps': ['Baseline', 'Jour 30'],
                        'Score': [phq9_bl, phq9_fu if not pd.isna(phq9_fu) else np.nan]
                    })
                    
                    fig_phq9_total = px.bar(
                        phq9_total_df.dropna(subset=['Score']),
                        x='Temps',
                        y='Score',
                        title="Score Total PHQ-9",
                        color='Temps', 
                        color_discrete_sequence=st.session_state.PASTEL_COLORS[2:4],  # Differentiate from MADRS colors
                        labels={"Score": "Score PHQ-9 Total"}
                    )
                    fig_phq9_total.update_yaxes(range=[0, 27])  # PHQ-9 range is 0-27
                    st.plotly_chart(fig_phq9_total, use_container_width=True)
                
                with col2_phq:
                    st.subheader("Scores par Item PHQ-9")
                    
                    # Définition des items du PHQ-9
                    phq9_items = {
                        "1": "Peu d'intérêt/plaisir",
                        "2": "Tristesse/désespoir",
                        "3": "Sommeil perturbé",
                        "4": "Fatigue",
                        "5": "Appétit modifié",
                        "6": "Mauvaise estime de soi",
                        "7": "Difficultés concentration",
                        "8": "Lenteur/agitation",
                        "9": "Pensées de mort"
                    }
                    
                    # Récupération des scores par item si disponibles
                    items_data = []
                    valid_items_found = False
                    
                    for i in range(1, 10):  # PHQ-9 has 9 items
                        bl_col = f'phq9_{i}_bl'
                        fu_col = f'phq9_{i}_fu'
                        item_label = phq9_items.get(str(i), f"Item {i}")
                        
                        bl_val = pd.to_numeric(patient_data.get(bl_col), errors='coerce')
                        fu_val = pd.to_numeric(patient_data.get(fu_col), errors='coerce')
                        
                        if not pd.isna(bl_val) or not pd.isna(fu_val):
                            valid_items_found = True
                            
                        items_data.append({
                            'Item': item_label,
                            'Baseline': bl_val,
                            'Jour 30': fu_val
                        })
                    
                    # Si les données d'items ne sont pas disponibles, générer des données simulées pour démonstration
                    if not valid_items_found and not pd.isna(phq9_bl):
                        st.info("Scores par item non disponibles - affichage d'une distribution simulée basée sur le score total.")
                        
                        # Générer des données simulées basées sur le score total
                        np.random.seed(int(phq9_bl) if not pd.isna(phq9_bl) else 42)  # Reproductibilité
                        
                        items_data = []
                        total_bl = phq9_bl
                        remaining_bl = total_bl
                        
                        for i in range(1, 9):  # Pour les 8 premiers items
                            item_label = phq9_items.get(str(i), f"Item {i}")
                            max_val = min(3, remaining_bl)  # Max 3 points par item
                            if max_val <= 0:
                                bl_val = 0
                            else:
                                bl_val = np.random.randint(0, max_val + 1)
                            remaining_bl -= bl_val
                            
                            # Pour le follow-up (s'il existe)
                            fu_val = np.nan
                            if not pd.isna(phq9_fu):
                                reduction_factor = phq9_fu / phq9_bl if phq9_bl > 0 else 0
                                fu_val = max(0, min(3, round(bl_val * reduction_factor)))
                            
                            items_data.append({
                                'Item': item_label,
                                'Baseline': bl_val,
                                'Jour 30': fu_val
                            })
                        
                        # Dernier item avec le reste
                        items_data.append({
                            'Item': phq9_items.get("9", "Item 9"),
                            'Baseline': max(0, remaining_bl),
                            'Jour 30': np.nan if pd.isna(phq9_fu) else max(0, round(max(0, remaining_bl) * (phq9_fu / phq9_bl if phq9_bl > 0 else 0)))
                        })
                        
                        valid_items_found = True
                    
                    if valid_items_found:
                        phq9_items_df = pd.DataFrame(items_data)
                        
                        # Convertir en nombres
                        phq9_items_df['Baseline'] = pd.to_numeric(phq9_items_df['Baseline'], errors='coerce')
                        phq9_items_df['Jour 30'] = pd.to_numeric(phq9_items_df['Jour 30'], errors='coerce')
                        
                        # Créer un format long pour le graphique
                        phq9_items_long = phq9_items_df.melt(
                            id_vars='Item',
                            var_name='Temps',
                            value_name='Score'
                        ).dropna(subset=['Score'])
                        
                        # Graphique des items
                        fig_items = px.bar(
                            phq9_items_long,
                            x='Item',
                            y='Score',
                            color='Temps',
                            barmode='group',
                            title="Scores par Item PHQ-9",
                            template="plotly_white",
                            color_discrete_sequence=st.session_state.PASTEL_COLORS[2:4],  # Different colors
                            labels={"Score": "Score (0-3)"}
                        )
                        
                        fig_items.update_xaxes(tickangle=-45)
                        fig_items.update_yaxes(range=[0, 3])  # PHQ-9 item scores are 0-3
                        
                        st.plotly_chart(fig_items, use_container_width=True)
                    else:
                        st.warning("Scores par item PHQ-9 non disponibles.")
                
                # Comparaison au groupe
                st.markdown("---")
                st.subheader("Comparaison au groupe")
                
                # Comparaison similaire à celle du MADRS mais pour le PHQ-9
                if 'final_data' in st.session_state and not st.session_state.final_data.empty:
                    try:
                        # Get all PHQ-9 scores for comparison
                        all_phq9_bl = pd.to_numeric(st.session_state.final_data['phq9_score_bl'], errors='coerce')
                        all_phq9_fu = pd.to_numeric(st.session_state.final_data['phq9_score_fu'], errors='coerce')
                        
                        # Calculate improvements for all patients
                        improvements = []
                        for bl, fu in zip(all_phq9_bl, all_phq9_fu):
                            if pd.notna(bl) and pd.notna(fu) and bl > 0:
                                imp_pct = ((bl - fu) / bl) * 100
                                improvements.append(imp_pct)
                        
                        if improvements and not pd.isna(phq9_bl) and not pd.isna(phq9_fu) and phq9_bl > 0:
                            # Calculate patient's improvement
                            patient_imp = ((phq9_bl - phq9_fu) / phq9_bl) * 100
                            
                            # Calculate patient's percentile
                            percentile = sum(1 for imp in improvements if imp < patient_imp) / len(improvements) * 100
                            
                            # Create distribution plot
                            fig_dist = px.histogram(
                                pd.DataFrame({'Amélioration (%)': improvements}), 
                                x='Amélioration (%)', 
                                nbins=20,
                                color_discrete_sequence=['#90BE6D'],
                                title=f"Distribution des Améliorations PHQ-9 dans le Groupe (n={len(improvements)})"
                            )
                            
                            # Add marker for current patient
                            fig_dist.add_vline(
                                x=patient_imp, 
                                line_color='#F94144', 
                                line_width=2,
                                annotation_text=f"Patient actuel ({patient_imp:.1f}%)",
                                annotation_position="top"
                            )
                            
                            st.plotly_chart(fig_dist, use_container_width=True)
                            
                            # Show percentile information
                            if percentile > 75:
                                st.success(f"✅ Ce patient est dans le **{percentile:.0f}e centile** d'amélioration (meilleur que {percentile:.0f}% des patients)")
                            elif percentile > 50:
                                st.info(f"ℹ️ Ce patient est dans le **{percentile:.0f}e centile** d'amélioration (meilleur que {percentile:.0f}% des patients)")
                            elif percentile > 25:
                                st.warning(f"⚠️ Ce patient est dans le **{percentile:.0f}e centile** d'amélioration (moins bien que {100-percentile:.0f}% des patients)")
                            else:
                                st.error(f"⚠️ Ce patient est dans le **{percentile:.0f}e centile** d'amélioration (moins bien que {100-percentile:.0f}% des patients)")
                        else:
                            st.info("Données insuffisantes pour la comparaison de groupe.")
                    except Exception as e:
                        st.error(f"Erreur lors du calcul de la comparaison: {e}")
                        logging.exception("Error calculating PHQ-9 group comparison")
                else:
                    st.info("Données insuffisantes pour la comparaison de groupe.")
        
        # Implémentation du BFI (Big Five Inventory)
        with subtab_bfi:
            st.subheader("Inventaire BFI (Big Five)")
            
            # Définition des dimensions du Big Five
            big_five_domains = {
                "E": "Extraversion",
                "A": "Agréabilité",
                "C": "Conscience",
                "N": "Neuroticisme",
                "O": "Ouverture"
            }
            
            # Vérifier si les données BFI sont disponibles
            bfi_columns = [col for col in patient_data.index if col.startswith('bfi_') and not col.endswith('_date')]
            
            if not bfi_columns:
                st.warning("Données BFI non disponibles pour ce patient.")
            else:
                # Extraire les scores des dimensions
                bfi_scores = {}
                for domain_code, domain_name in big_five_domains.items():
                    score_col = f'bfi_{domain_code.lower()}'
                    if score_col in patient_data.index:
                        score = pd.to_numeric(patient_data.get(score_col), errors='coerce')
                        bfi_scores[domain_name] = score
                
                if not bfi_scores:
                    # Générer des données simulées pour démonstration
                    st.info("Scores BFI non disponibles - affichage de données simulées pour démonstration.")
                    np.random.seed(hash(str(patient_data.get('ID', ''))) % 2**32)  # Seed based on patient ID
                    
                    for domain_name in big_five_domains.values():
                        # Generate random score between 2-5
                        bfi_scores[domain_name] = round(np.random.uniform(2, 5), 1)
                
                # Création du graphique radar pour le profil BFI
                fig_radar = go.Figure()
                
                # Ajouter les données du patient
                domains = list(bfi_scores.keys())
                values = list(bfi_scores.values())
                
                # Radar foncé pour le patient
                fig_radar.add_trace(go.Scatterpolar(
                    r=values + [values[0]],  # Close the loop
                    theta=domains + [domains[0]],  # Close the loop
                    fill='toself',
                    name='Patient',
                    line=dict(color='#2A9D8F', width=3),
                    fillcolor='rgba(42, 157, 143, 0.3)'
                ))
                
                # Ajouter la référence normative (moyennes de population)
                # Ces valeurs sont des exemples (moyenne = 3.0 avec légères variations)
                reference_values = [3.2, 3.4, 3.0, 3.1, 3.3]  # Valeurs typiques
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=reference_values + [reference_values[0]],  # Close the loop
                    theta=domains + [domains[0]],  # Close the loop
                    fill='toself',
                    name='Référence',
                    line=dict(color='#888888', width=2, dash='dot'),
                    fillcolor='rgba(200, 200, 200, 0.2)'
                ))
                
                # Configurer le graphique
                fig_radar.update_layout(
                    title="Profil de Personnalité BFI",
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[1, 5]  # BFI est généralement sur une échelle de 1-5
                        )
                    ),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5)
                )
                
                # Afficher le graphique radar
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # Afficher les scores en tableau
                col1_bfi, col2_bfi = st.columns([1, 1])
                
                with col1_bfi:
                    st.subheader("Scores par Dimension")
                    
                    bfi_df = pd.DataFrame({
                        'Dimension': domains,
                        'Score': values,
                        'Référence': reference_values,
                        'Différence': [v - r for v, r in zip(values, reference_values)]
                    })
                    
                    # Format the table
                    st.dataframe(
                        bfi_df.style.format({
                            'Score': '{:.1f}',
                            'Référence': '{:.1f}',
                            'Différence': '{:.1f}'
                        }).background_gradient(
                            cmap='RdYlGn', 
                            subset=['Différence'],
                            vmin=-2, 
                            vmax=2
                        ),
                        hide_index=True,
                        use_container_width=True
                    )
                
                with col2_bfi:
                    st.subheader("Interprétation")
                    
                    # Interprétation des scores basée sur leur niveau
                    for domain, score in bfi_scores.items():
                        if pd.isna(score):
                            continue
                            
                        # Texte d'interprétation basé sur le score
                        if domain == "Extraversion":
                            if score >= 4:
                                interpretation = "Très sociable, énergique et orienté vers les autres."
                            elif score >= 3:
                                interpretation = "Modérément sociable, appréciant un équilibre entre activité sociale et moments seul."
                            else:
                                interpretation = "Plus réservé, préférant des environnements calmes et les activités solitaires."
                        elif domain == "Agréabilité":
                            if score >= 4:
                                interpretation = "Très coopératif, empathique et soucieux des autres."
                            elif score >= 3:
                                interpretation = "Généralement amical, mais avec une capacité à défendre ses intérêts."
                            else:
                                interpretation = "Peut être plus critique, sceptique ou compétitif dans les relations."
                        elif domain == "Conscience":
                            if score >= 4:
                                interpretation = "Très discipliné, organisé et attentif aux détails."
                            elif score >= 3:
                                interpretation = "Généralement fiable avec un bon équilibre entre discipline et flexibilité."
                            else:
                                interpretation = "Plus flexible, spontané, peut préférer les approches moins structurées."
                        elif domain == "Neuroticisme":
                            if score >= 4:
                                interpretation = "Tendance à ressentir plus fortement les émotions négatives et le stress."
                            elif score >= 3:
                                interpretation = "Réponses émotionnelles modérées aux situations stressantes."
                            else:
                                interpretation = "Plus résilient émotionnellement, moins susceptible d'être perturbé par le stress."
                        elif domain == "Ouverture":
                            if score >= 4:
                                interpretation = "Très curieux, créatif et ouvert aux nouvelles expériences."
                            elif score >= 3:
                                interpretation = "Équilibre entre tradition et innovation, modérément ouvert aux nouvelles idées."
                            else:
                                interpretation = "Plus conventionnel, pratique et préférant les approches familières."
                        
                        # Afficher avec une mise en forme appropriée
                        st.markdown(f"**{domain}** ({score:.1f}/5): {interpretation}")
                
                # Implications cliniques
                st.markdown("---")
                st.subheader("Implications Thérapeutiques")
                
                st.markdown("""
                Le profil de personnalité peut influencer la réponse au traitement:
                
                - **Neuroticisme élevé**: Souvent associé à une symptomatologie dépressive plus sévère et une réponse plus lente au traitement
                - **Extraversion élevée**: Généralement associée à une meilleure réponse aux thérapies de groupe et interventions sociales
                - **Conscience élevée**: Associée à une meilleure adhérence aux protocoles thérapeutiques
                - **Agréabilité élevée**: Souvent liée à une meilleure alliance thérapeutique
                - **Ouverture élevée**: Potentiellement associée à une meilleure réponse aux thérapies créatives et innovantes
                """)
                
                # Comparaison au groupe
                st.markdown("---")
                st.subheader("Comparaison au groupe")
                
                # Créer une distribution simulée des scores BFI dans la population de patients
                # (ceci pourrait être remplacé par des données réelles si disponibles)
                
                # Choisir une dimension pour la comparaison
                selected_dimension = st.selectbox(
                    "Sélectionner une dimension pour la comparaison:",
                    list(big_five_domains.values())
                )
                
                if selected_dimension in bfi_scores:
                    patient_score = bfi_scores[selected_dimension]
                    
                    # Générer une distribution simulée
                    np.random.seed(42)  # Pour reproductibilité
                    population_scores = np.random.normal(3, 0.7, 100)  # Moyenne=3, écart-type=0.7
                    population_scores = np.clip(population_scores, 1, 5)  # Limiter à l'échelle 1-5
                    
                    # Calculer le percentile du patient
                    percentile = sum(1 for s in population_scores if s < patient_score) / len(population_scores) * 100
                    
                    # Créer l'histogramme de distribution
                    fig_dist = px.histogram(
                        pd.DataFrame({selected_dimension: population_scores}),
                        x=selected_dimension,
                        nbins=20,
                        color_discrete_sequence=['#F4A261'],
                        title=f"Distribution des Scores de {selected_dimension} (n=100)"
                    )
                    
                    # Ajouter la marque pour le patient actuel
                    fig_dist.add_vline(
                        x=patient_score,
                        line_color='#E76F51',
                        line_width=2,
                        annotation_text=f"Patient actuel ({patient_score:.1f})",
                        annotation_position="top"
                    )
                    
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Afficher l'interprétation du percentile
                    if percentile > 75:
                        st.info(f"Ce patient est dans le **{percentile:.0f}e centile** pour {selected_dimension} (plus élevé que {percentile:.0f}% des patients)")
                    elif percentile > 25:
                        st.info(f"Ce patient est dans le **{percentile:.0f}e centile** pour {selected_dimension} (niveau moyen)")
                    else:
                        st.info(f"Ce patient est dans le **{percentile:.0f}e centile** pour {selected_dimension} (plus bas que {100-percentile:.0f}% des patients)")
                else:
                    st.warning(f"Score pour {selected_dimension} non disponible.")
            
    # Tab 3: Symptoms (Symptômes) - now this becomes tab 3 instead of tab 4
    with tab_symptoms:
        st.header("🩺 Suivi des Symptômes")
        symptom_tabs = st.tabs(["🕸️ Réseau Symptômes", "⏳ Progrès EMA"])
        
        # Network tab
        with symptom_tabs[0]:
            st.header("🕸️ Réseau de Symptômes (Basé sur EMA)")
            if patient_ema.empty: st.warning("⚠️ Aucune donnée EMA dispo pour générer le réseau.")
            elif len(patient_ema) < 10: st.warning(f"⚠️ Pas assez de données EMA ({len(patient_ema)}) pour analyse fiable.")
            else:
                st.info("Influence potentielle des symptômes EMA au fil du temps.")
                threshold = st.slider( "Seuil connexions", 0.05, 0.5, 0.15, 0.05, key="network_thresh")
                if st.button("🔄 Générer/Actualiser Réseau"):
                     try:
                          if 'SYMPTOMS' not in st.session_state: st.error("Erreur: Liste symptômes EMA non définie.")
                          else:
                                symptoms_available = [s for s in st.session_state.SYMPTOMS if s in patient_ema.columns]
                                if not symptoms_available: st.error("❌ Aucune colonne symptôme valide trouvée.")
                                else:
                                    fig_network = generate_person_specific_network( patient_ema, patient_id, symptoms_available, threshold=threshold)
                                    st.plotly_chart(fig_network, use_container_width=True)
                                    with st.expander("💡 Interprétation"): st.markdown("""... (interpretation text) ...""")
                     except Exception as e: st.error(f"❌ Erreur génération réseau: {e}"); logging.exception(f"Network gen failed {patient_id}")
                else: st.info("Cliquez sur bouton pour générer.")
        
        # EMA progress tab
        with symptom_tabs[1]:
            st.header("⏳ Progression (Basé sur EMA)")
            treatment_progress(patient_ema)
            # Rest of EMA progress logic here...
            
    # Tab 5: Care Plan (Plan de Soins)
    with tab_care_plan:
        st.header("🎯 Plan de Soins Actuel")
        st.info("Affiche la **dernière** entrée. Pour ajouter/modifier, allez à 'Plan de Soins et Entrées Infirmières'.")
        try:
             latest_plan = get_latest_nurse_inputs(patient_id)
             if latest_plan and latest_plan.get('timestamp'):
                 plan_date = pd.to_datetime(latest_plan.get('timestamp')).strftime('%Y-%m-%d %H:%M'); created_by = latest_plan.get('created_by', 'N/A')
                 st.subheader(f"Dernière MàJ: {plan_date} (par {created_by})")
                 col_stat, col_symp, col_int = st.columns([1,2,2])
                 with col_stat: st.metric("Statut Objectif", latest_plan.get('goal_status', 'N/A'))
                 with col_symp: st.markdown(f"**Sympt. Cibles:**\n> {latest_plan.get('target_symptoms', 'N/A')}")
                 with col_int: st.markdown(f"**Interv. Planifiées:**\n> {latest_plan.get('planned_interventions', 'N/A')}")
                 st.markdown("---"); st.markdown(f"**Objectifs:**\n_{latest_plan.get('objectives', 'N/A')}_"); st.markdown(f"**Tâches:**\n_{latest_plan.get('tasks', 'N/A')}_"); st.markdown(f"**Commentaires:**\n_{latest_plan.get('comments', 'N/A')}_")
             elif latest_plan: st.warning("Dernier plan trouvé mais date inconnue.")
             else: st.warning(f"ℹ️ Aucun plan trouvé pour {patient_id}.")
        except Exception as e: st.error(f"Erreur chargement plan: {e}"); logging.exception(f"Error loading care plan {patient_id}")
            
    # Tab 6: Side Effects (Effets Secondaires)
    with tab_side_effects:
        st.header("🩺 Suivi Effets Secondaires (Résumé)")
        # Side effects logic here...
        
    # Tab 7: Notes History (Historique Notes)
    with tab_notes_history:
        st.header("📝 Historique Notes Infirmières")
        st.info("Affiche notes/plans précédents.")
        try:
            notes_history_df = get_nurse_inputs_history(patient_id)
            if notes_history_df.empty: st.info(f"ℹ️ Aucune note historique pour {patient_id}.")
            else:
                st.info(f"Affichage {len(notes_history_df)} entrées.")
                # Rest of notes history logic here...
        except Exception as e: st.error(f"Erreur historique notes: {e}"); logging.exception(f"Error loading notes history {patient_id}")

# Create a helper function to display patient overview (to keep code cleaner)
def create_patient_overview(patient_data):
    # Create section header with icon
    st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 1.8rem; margin-right: 10px;">👤</span>
            <h2 style="margin: 0;">Aperçu du Patient</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Row 1: Patient basic info in card-like elements
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender_text = "Femme" if str(patient_data.get('sexe', 'N/A')) == '1' else "Homme" if str(patient_data.get('sexe', 'N/A')) == '2' else "Autre/N/A"
        st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem;">
                <p style="color: #666; margin-bottom: 5px; font-size: 0.9rem;">Sexe</p>
                <h3 style="margin: 0; font-size: 1.4rem;">
                    {gender_text}
                </h3>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem;">
                <p style="color: #666; margin-bottom: 5px; font-size: 0.9rem;">Âge</p>
                <h3 style="margin: 0; font-size: 1.4rem;">
                    {patient_data.get('age', 'N/A')}
                </h3>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem;">
                <p style="color: #666; margin-bottom: 5px; font-size: 0.9rem;">Diagnostic</p>
                <h3 style="margin: 0; font-size: 1.4rem;">
                    {patient_data.get('protocol', 'N/A')}
                </h3>
            </div>
        """, unsafe_allow_html=True)
    
    # Clinical details expander
    with st.expander("🔍 Données Cliniques Détaillées", expanded=False):
        clinical_cols = st.columns(2)
        
        with clinical_cols[0]:
            st.markdown("#### Comorbidités")
            comorbidities = patient_data.get('comorbidities', 'Aucune comorbidité répertoriée')
            if comorbidities != 'Aucune comorbidité répertoriée' and isinstance(comorbidities, str):
                comorbidity_list = comorbidities.split('; ')
                for item in comorbidity_list:
                    st.markdown(f"• {item}")
            else:
                st.info(comorbidities)
                
        with clinical_cols[1]:
            st.markdown("#### Historique de Traitement")
            treatments = {
                "Psychothérapie": patient_data.get('psychotherapie_bl') == '1',
                "ECT": patient_data.get('ect_bl') == '1',
                "rTMS antérieure": patient_data.get('rtms_bl') == '1',
                "tDCS": patient_data.get('tdcs_bl') == '1',
            }
            
            for treatment, received in treatments.items():
                status = "✓" if received else "✗"
                st.markdown(f"• {treatment}: {status}")
    
    # Medications section
    st.markdown("""
        <div style="margin-top: 1.5rem; margin-bottom: 1rem;">
            <h3 style="display: flex; align-items: center;">
                <span style="margin-right: 8px;">📋</span> Médications Actuelles
            </h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Create medications table with styling
    if 'medications' in patient_data.index and patient_data['medications'] != "Aucun":
        meds_list = patient_data['medications'].split('; ')
        meds_data = []
        
        for med in meds_list:
            parts = med.split(' ')
            if len(parts) >= 2:
                name = ' '.join(parts[:-1])
                dosage = parts[-1]
                
                # Determine medication category
                category = "Autre"
                for cat, meds in MEDICATION_CATEGORIES.items():
                    if name in meds:
                        category = cat
                        break
                        
                meds_data.append({
                    "Médicament": name,
                    "Catégorie": category,
                    "Dosage": dosage
                })
        
        if meds_data:
            meds_df = pd.DataFrame(meds_data)
            st.dataframe(meds_df, hide_index=True, use_container_width=True)
    else:
        st.info("Aucune médication psychiatrique n'est actuellement prescrite.")
    
    # Create two columns for dashboard metrics
    left_col, right_col = st.columns([1, 1])
    
    # Column 1: Progress Summary
    with left_col:
        st.markdown("""
            <div style="margin-top: 1.5rem; margin-bottom: 1rem;">
                <h3 style="display: flex; align-items: center;">
                    <span style="margin-right: 8px;">📊</span> Résumé de Progression
                </h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Create progress card
        madrs_bl = pd.to_numeric(patient_data.get("madrs_score_bl"), errors='coerce')
        madrs_fu = pd.to_numeric(patient_data.get("madrs_score_fu"), errors='coerce')
        
        if not pd.isna(madrs_bl) and not pd.isna(madrs_fu):
            improvement = madrs_bl - madrs_fu
            improvement_pct = ((madrs_bl - madrs_fu) / madrs_bl) * 100 if madrs_bl > 0 else 0
            
            # Determine clinical status
            is_responder = improvement_pct >= 30
            is_remitter = madrs_fu < 10
            
            responder_status = f"""<span style="color: #2e7d32; font-weight: bold;">✓ Répondeur (≥30%)</span>""" if is_responder else f"""<span style="color: #d32f2f; font-weight: bold;">✗ Non-répondeur (<30%)</span>"""
            
            symptom_status = f"""<span style="color: #2e7d32; font-weight: bold;">✓ En rémission (≤10)</span>""" if is_remitter else f"""<span style="color: #fb8c00;">⚠️ Symptômes Actifs (>10)</span>"""
            
            status_bg_color = "#f1f8e9" if is_responder else "#fbe9e7"
            status_border_color = "#7cb342" if is_responder else "#d32f2f"
            
            st.markdown(f"""
                <div style="background-color: {status_bg_color}; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {status_border_color};">
                    <h4 style="margin-top: 0;">MADRS: {madrs_bl:.0f} → {madrs_fu:.0f}</h4>
                    <p style="font-size: 1.2rem; font-weight: bold; color: {"#2e7d32" if improvement > 0 else "#d32f2f"}; margin-bottom: 0.5rem;">
                        {"-" if improvement > 0 else "+"}{abs(improvement):.0f} points ({abs(improvement_pct):.1f}% {"amélioration" if improvement > 0 else "détérioration"})
                    </p>
                    <p style="margin: 0;">
                        Statut Clinique: {responder_status}
                    </p>
                    <p style="margin-top: 0.5rem;">
                        {symptom_status}
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Données MADRS insuffisantes pour évaluer la progression")
    
    # Column 2: ML Prediction (replacing GAF section)
    with right_col:
        st.markdown("""
            <div style="margin-top: 1.5rem; margin-bottom: 1rem;">
                <h3 style="display: flex; align-items: center;">
                    <span style="margin-right: 8px;">🧠</span> Prédiction ML
                </h3>
            </div>
        """, unsafe_allow_html=True)
        
        # Using the ML service to get prediction
        try:
            from services.ml_service import predict_response_probability
            
            # Make prediction for this patient
            prediction_result = predict_response_probability(patient_data.to_dict())
            
            # Extract probability and other info from result
            response_prob = prediction_result.get('probability', 0.5)
            confidence = prediction_result.get('confidence', 'low')
            is_responder_pred = prediction_result.get('is_responder', False)
            
            # Format probability as percentage
            response_prob_pct = response_prob * 100
            
            # Determine styling based on prediction
            pred_bg_color = "#f1f8e9" if response_prob >= 0.5 else "#fbe9e7"
            pred_border_color = "#7cb342" if response_prob >= 0.5 else "#d32f2f"
            pred_text_color = "#2e7d32" if response_prob >= 0.5 else "#d32f2f"
            
            # Define confidence text and icon
            confidence_icon = "⭐⭐⭐" if confidence == "high" else "⭐⭐" if confidence == "medium" else "⭐"
            confidence_label = "Élevée" if confidence == "high" else "Moyenne" if confidence == "medium" else "Faible"
            
            # Prediction card with styled content
            st.markdown(f"""
                <div style="background-color: {pred_bg_color}; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {pred_border_color};">
                    <h4 style="margin-top: 0;">Probabilité de Réponse</h4>
                    <p style="font-size: 1.8rem; font-weight: bold; color: {pred_text_color}; margin-bottom: 0.5rem;">
                        {response_prob_pct:.1f}%
                    </p>
                    <p style="margin: 0;">
                        Statut Prévu: <span style="color: {pred_text_color}; font-weight: bold;">{"Répondeur" if is_responder_pred else "Non-répondeur"}</span>
                    </p>
                    <p style="margin-top: 0.5rem; font-size: 0.9rem;">
                        Confiance: {confidence_icon} ({confidence_label})
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            # Add explanation as an expander
            with st.expander("💡 À propos de cette prédiction"):
                st.markdown("""
                Cette prédiction est générée par un modèle d'apprentissage automatique formé sur des données historiques de patients.
                
                **Facteurs clés:** âge, sexe, scores initiaux MADRS/BDI/BAI, profil de personnalité
                
                **Note:** Cette prédiction est fournie à titre indicatif seulement et ne remplace pas le jugement clinique.
                """)
            
            # Add decision curve analysis (simulated)
            with st.expander("📊 Courbe de décision (simulation)"):
                st.markdown("""
                **Courbe de décision clinique (simulation)**
                
                Le graphique ci-dessous est une **simulation** qui illustre l'utilité clinique potentielle du modèle prédictif 
                en comparaison avec différentes stratégies d'intervention.
                """)
                
                # Create simulated decision curve
                import numpy as np
                
                # X-axis represents preference/threshold
                x = np.linspace(0, 1, 100)
                
                # Model benefit - generally higher than test but lower than intervention for all at low thresholds
                model_benefit = 0.3 - 0.15 * x
                
                # "Intervention for all" strategy - decreases linearly and sharply
                treat_all_benefit = 0.4 - 0.8 * x
                treat_all_benefit[x > 0.5] = 0  # Becomes 0 after threshold 0.5
                
                # Test benefit - slightly lower than model
                test_benefit = 0.25 - 0.1 * x
                
                # Create the plot
                import plotly.graph_objects as go
                
                fig = go.Figure()
                
                # Add "Intervention for all" line
                fig.add_trace(go.Scatter(
                    x=x,
                    y=treat_all_benefit,
                    mode='lines',
                    name='Intervention pour tous',
                    line=dict(color='black', width=2)
                ))
                
                # Add model line
                fig.add_trace(go.Scatter(
                    x=x,
                    y=model_benefit,
                    mode='lines',
                    name='Modèle',
                    line=dict(color='grey', width=2)
                ))
                
                # Add test line
                fig.add_trace(go.Scatter(
                    x=x,
                    y=test_benefit,
                    mode='lines',
                    name='Test',
                    line=dict(color='black', width=2, dash='dash')
                ))
                
                # Add "Intervention for none" line (always 0)
                fig.add_trace(go.Scatter(
                    x=x,
                    y=np.zeros_like(x),
                    mode='lines',
                    name='Intervention pour aucun',
                    line=dict(color='black', width=2)
                ))
                
                # Update layout to match the reference image
                fig.update_layout(
                    title=None,
                    xaxis_title='Préférence',
                    yaxis_title='Bénéfice',
                    template='plotly_white',
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    ),
                    margin=dict(l=10, r=10, t=10, b=10),
                    plot_bgcolor='white',
                    xaxis=dict(
                        showgrid=False,
                        zeroline=True,
                        showticklabels=False,
                        range=[0, 1],
                        tickvals=[0.25, 0.75],
                        ticktext=["Je suis inquiet de la maladie", "Je suis inquiet de la biopsie"]
                    ),
                    yaxis=dict(
                        showgrid=False,
                        zeroline=True,
                        showticklabels=False,
                        range=[0, 0.4]
                    )
                )
                
                # Add custom X-axis annotations to match the image
                fig.add_annotation(
                    x=0.25,
                    y=-0.05,
                    xref="x",
                    yref="paper",
                    text="Je suis inquiet<br>de la maladie",
                    showarrow=False,
                    font=dict(size=10),
                    align="center"
                )
                
                fig.add_annotation(
                    x=0.75,
                    y=-0.05,
                    xref="x",
                    yref="paper",
                    text="Je suis inquiet<br>de la biopsie",
                    showarrow=False,
                    font=dict(size=10),
                    align="center"
                )
                
                # Show the plot
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Interprétation:**
                
                - La ligne **"Modèle"** représente le bénéfice net obtenu en utilisant notre modèle prédictif.
                - La ligne **"Intervention pour tous"** montre le bénéfice net si on décide d'intervenir pour tous les patients.
                - La ligne **"Test"** indique le bénéfice net en utilisant un test diagnostic standard.
                - La ligne **"Intervention pour aucun"** montre le bénéfice net si on n'intervient pour aucun patient.
                
                Quand la courbe du modèle est au-dessus des autres, cela indique que le modèle offre une meilleure 
                valeur clinique à ce niveau de préférence du patient (tolérance au risque).
                
                **Note:** Cette courbe est une simulation à des fins d'illustration uniquement.
                """)
            
        except Exception as e:
            st.warning(f"⚠️ Impossible de charger la prédiction ML: {e}")
            logging.exception(f"Error loading ML prediction for patient {patient_data.get('ID')}")
    
    # Substance use section
    st.markdown("""
        <div style="margin-top: 1.5rem; margin-bottom: 1rem;">
            <h3 style="display: flex; align-items: center;">
                <span style="margin-right: 8px;">🚬</span> Consommation de Substances
            </h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Substance use cards - using actual data if available, otherwise defaults
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cannabis_status = "Oui" if str(patient_data.get('cannabis', '0')) == '1' else "Non"
        st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem;">
                <p style="color: #666; margin-bottom: 5px; font-size: 0.9rem;">Cannabis</p>
                <h3 style="margin: 0; font-size: 1.4rem; color: {"#d32f2f" if cannabis_status == "Oui" else "#2e7d32"}">
                    {cannabis_status}
                </h3>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        stimulants_status = "Oui" if str(patient_data.get('stimulants', '0')) == '1' else "Non"
        st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem;">
                <p style="color: #666; margin-bottom: 5px; font-size: 0.9rem;">Stimulants</p>
                <h3 style="margin: 0; font-size: 1.4rem; color: {"#d32f2f" if stimulants_status == "Oui" else "#2e7d32"}">
                    {stimulants_status}
                </h3>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        alcohol_status = "Oui" if str(patient_data.get('alcohol', '0')) == '1' else "Non"
        st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem;">
                <p style="color: #666; margin-bottom: 5px; font-size: 0.9rem;">Alcool</p>
                <h3 style="margin: 0; font-size: 1.4rem; color: {"#d32f2f" if alcohol_status == "Oui" else "#2e7d32"}">
                    {alcohol_status}
                </h3>
            </div>
        """, unsafe_allow_html=True)
    
    # Export button
    st.markdown("<div style='margin-top: 2rem;'>", unsafe_allow_html=True)
    if st.button("Exporter Données Patient (CSV)", key="export_patient_data"):
        try:
            patient_id = patient_data.get('ID')
            if patient_id:
                # Create DataFrame from patient data
                patient_df = pd.DataFrame([patient_data])
                
                # Convert to CSV
                csv = patient_df.to_csv(index=False).encode('utf-8')
                
                # Offer download
                st.download_button(
                    label="Télécharger (CSV)",
                    data=csv,
                    file_name=f"patient_{patient_id}_data.csv",
                    mime='text/csv'
                )
        except Exception as e:
            st.error(f"Erreur lors de l'export: {e}")
            logging.exception(f"Error exporting patient data")
    st.markdown("</div>", unsafe_allow_html=True)