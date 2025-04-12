# components/overview.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def main_dashboard_page():
    """Main overview dashboard with key metrics"""
    # Apply custom styling
    st.markdown("""
    <style>
        /* Modern, flat design without shadows */
        .overview-header {
            font-size: 1.8rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: #333;
        }
        
        /* Metric panels with flat design */
        .metric-panel {
            background-color: #f8fafc;
            border-radius: 0.5rem;
            padding: 1.25rem;
            border: 1px solid #edf2f7;
            margin-bottom: 1.5rem;
        }
        
        /* Clean metric display */
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #333;
            margin: 0;
            line-height: 1.2;
        }
        
        .metric-label {
            color: #64748b;
            font-size: 0.9rem;
            font-weight: 500;
            margin-bottom: 0.25rem;
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.25rem;
            font-weight: 600;
            margin: 1.5rem 0 1rem 0;
            color: #333;
        }
        
        /* Tab styling */
        .custom-tab {
            display: inline-flex;
            align-items: center;
            padding: 0.5rem 1rem;
            border-bottom: 3px solid transparent;
            font-weight: 500;
            font-size: 0.9rem;
            color: #64748b;
            margin-right: 0.5rem;
        }
        
        .custom-tab.active {
            border-bottom-color: #3b82f6;
            color: #1e40af;
        }
        
        .tab-container {
            display: flex;
            border-bottom: 1px solid #e2e8f0;
            margin-bottom: 1.5rem;
            overflow-x: auto;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Page header with cleaner design
    st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
            <h1 style="margin: 0; font-size: 1.8rem; font-weight: 600; color: #333;">Vue d'Ensemble</h1>
        </div>
    """, unsafe_allow_html=True)

    # Patient selection with cleaner design
    if hasattr(st.session_state, 'final_data') and not st.session_state.final_data.empty:
        # Create a container with shadow-free styling
        st.markdown("""
            <div style="
                background-color: #f8fafc; 
                border-radius: 0.5rem; 
                padding: 0.75rem; 
                border: 1px solid #edf2f7;
                margin-bottom: 1.5rem;
            ">
                <div style="font-size: 0.9rem; font-weight: 500; color: #64748b; margin-bottom: 0.5rem;">
                    Sélectionner un patient:
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Get patient IDs and add selection dropdown inside
        all_patient_ids = sorted(st.session_state.final_data['ID'].unique().tolist())
        if all_patient_ids:
            sel_col, btn_col = st.columns([3, 1])
            with sel_col:
                selected_patient = st.selectbox(
                    "Sélectionner un patient:",
                    all_patient_ids,
                    index=0 if st.session_state.selected_patient_id not in all_patient_ids else all_patient_ids.index(st.session_state.selected_patient_id),
                    key="overview_patient_selector",
                    label_visibility="collapsed"
                )
            with btn_col:
                if st.button("Voir détails", type="primary", key="view_details_btn"):
                    st.session_state.selected_patient_id = selected_patient
                    st.session_state.sidebar_selection = "Tableau de Bord du Patient"
                    st.rerun()
    
    # Display error if no data
    if not hasattr(st.session_state, 'final_data') or st.session_state.final_data.empty:
        st.error("Aucune donnée patient chargée.")
        return
    
    # Top metrics section with flat design
    st.markdown("<div style='margin-bottom: 2rem;'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        # Count total patients
        total_patients = len(st.session_state.final_data)
        st.markdown(f"""
            <div class="metric-panel">
                <div class="metric-label">Nombre Total de Patients</div>
                <div class="metric-value">{total_patients}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        # Calculate average MADRS improvement
        madrs_df = st.session_state.final_data[
            st.session_state.final_data['madrs_score_bl'].notna() & 
            st.session_state.final_data['madrs_score_fu'].notna()
        ]
        
        if not madrs_df.empty:
            improvement = madrs_df['madrs_score_bl'] - madrs_df['madrs_score_fu']
            avg_improvement = improvement.mean()
            st.markdown(f"""
                <div class="metric-panel">
                    <div class="metric-label">Amélioration MADRS Moyenne</div>
                    <div class="metric-value">{avg_improvement:.1f} points</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="metric-panel">
                    <div class="metric-label">Amélioration MADRS Moyenne</div>
                    <div class="metric-value">N/A</div>
                </div>
            """, unsafe_allow_html=True)

    with col3:
        # Calculate response rate (>= 50% improvement)
        if not madrs_df.empty:
            percent_improvement = (improvement / madrs_df['madrs_score_bl']) * 100
            response_rate = (percent_improvement >= 50).mean() * 100
            st.markdown(f"""
                <div class="metric-panel">
                    <div class="metric-label">Taux de Réponse</div>
                    <div class="metric-value">{response_rate:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="metric-panel">
                    <div class="metric-label">Taux de Réponse</div>
                    <div class="metric-value">N/A</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    
    # Create a two-column layout for main content
    left_col, right_col = st.columns([3, 2], gap="large")
    
    # Main sections with cleaner headers
    with left_col:
        st.markdown("<div class='section-header'>Progression des Patients</div>", unsafe_allow_html=True)
        
        if 'madrs_score_bl' in st.session_state.final_data.columns and 'madrs_score_fu' in st.session_state.final_data.columns:
            # Create a waterfall chart to show overall improvement
            madrs_scores = st.session_state.final_data[['ID', 'madrs_score_bl', 'madrs_score_fu']].dropna()
            
            if not madrs_scores.empty:
                # Prepare data for visualization
                madrs_scores['improvement'] = madrs_scores['madrs_score_bl'] - madrs_scores['madrs_score_fu']
                madrs_scores['improvement_pct'] = (madrs_scores['improvement'] / madrs_scores['madrs_score_bl'] * 100).round(1)
                madrs_scores = madrs_scores.sort_values('improvement_pct', ascending=False)
                
                # Add threshold lines for response and remission
                madrs_scores_sorted = madrs_scores.sort_values('ID')
                
                fig_before_after = go.Figure()
                fig_before_after.add_trace(go.Scatter(
                    x=madrs_scores_sorted['ID'],
                    y=madrs_scores_sorted['madrs_score_bl'],
                    mode='lines+markers',
                    name='Baseline',
                    line=dict(color=st.session_state.PASTEL_COLORS[0], width=2)
                ))
                fig_before_after.add_trace(go.Scatter(
                    x=madrs_scores_sorted['ID'],
                    y=madrs_scores_sorted['madrs_score_fu'],
                    mode='lines+markers',
                    name='Jour 30',
                    line=dict(color=st.session_state.PASTEL_COLORS[1], width=2)
                ))
                
                # Add threshold lines
                fig_before_after.add_shape(
                    type="line", line=dict(dash='dash', color='green', width=2),
                    x0=0, x1=1, xref="paper",
                    y0=10, y1=10, yref="y"
                )
                fig_before_after.add_annotation(
                    xref="paper", yref="y",
                    x=0.01, y=10,
                    text="Seuil de rémission (10)",
                    showarrow=False,
                    font=dict(color="green")
                )
                
                # For the main MADRS chart - cleaner minimal styling
                fig_before_after.update_layout(
                    title=None,  # Remove title and add as header above
                    xaxis_title="Patient ID",
                    yaxis_title="Score MADRS",
                    margin=dict(l=40, r=40, t=20, b=40),  # Reduced top margin
                    height=350,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#333"),
                    xaxis=dict(gridcolor='#f0f0f0'),
                    yaxis=dict(gridcolor='#f0f0f0')
                )
                
                st.plotly_chart(fig_before_after, use_container_width=True)
            else:
                st.warning("Données MADRS insuffisantes pour l'analyse.")
        else:
            st.warning("Les colonnes MADRS n'existent pas dans les données.")
    
    with right_col:
        # Résumé de Progression section with cleaner styling
        st.markdown("<div class='section-header'>Résumé de Progression</div>", unsafe_allow_html=True)
        
        if 'madrs_score_bl' in st.session_state.final_data.columns and 'madrs_score_fu' in st.session_state.final_data.columns:
            madrs_scores = st.session_state.final_data[['ID', 'madrs_score_bl', 'madrs_score_fu']].dropna()
            
            if not madrs_scores.empty:
                # Calculate improvement
                madrs_scores['improvement'] = madrs_scores['madrs_score_bl'] - madrs_scores['madrs_score_fu']
                madrs_scores['improvement_pct'] = (madrs_scores['improvement'] / madrs_scores['madrs_score_bl'] * 100)
                
                # Calculate key metrics
                total_analyzed = len(madrs_scores)
                responders = (madrs_scores['improvement_pct'] >= 50).sum()
                remission = (madrs_scores['madrs_score_fu'] <= 10).sum()
                
                # Create clean metrics panels
                st.markdown(f"""
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                        <div class="metric-panel" style="margin-bottom: 0;">
                            <div class="metric-label">Patients analysés</div>
                            <div class="metric-value" style="font-size: 1.8rem;">{total_analyzed}</div>
                        </div>
                        <div class="metric-panel" style="margin-bottom: 0;">
                            <div class="metric-label">Non-répondeurs</div>
                            <div class="metric-value" style="font-size: 1.8rem;">{total_analyzed - responders} ({(total_analyzed - responders)/total_analyzed:.1%})</div>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                        <div class="metric-panel" style="margin-bottom: 0;">
                            <div class="metric-label">Répondeurs (≥50%)</div>
                            <div class="metric-value" style="font-size: 1.8rem;">{responders} ({responders/total_analyzed:.1%})</div>
                        </div>
                        <div class="metric-panel" style="margin-bottom: 0;">
                            <div class="metric-label">En rémission (≤10)</div>
                            <div class="metric-value" style="font-size: 1.8rem;">{remission} ({remission/total_analyzed:.1%})</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Add a small chart showing the distribution of response categories
                response_data = {
                    'Catégorie': ['Rémission', 'Répondeurs', 'Non-répondeurs'],
                    'Nombre': [remission, responders - remission, total_analyzed - responders]
                }
                response_df = pd.DataFrame(response_data)
                
                # Update chart styling
                fig = px.bar(
                    response_df, 
                    y='Catégorie', 
                    x='Nombre',
                    orientation='h',
                    color='Catégorie',
                    color_discrete_sequence=[
                        '#a5b4fc',  # Light purple for Remission
                        '#93c5fd',  # Light blue for Responders
                        '#fca5a5',  # Light red for Non-responders
                    ],
                    height=200
                )
                fig.update_layout(
                    margin=dict(l=40, r=10, t=10, b=20), 
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='#f0f0f0'),
                    yaxis=dict(gridcolor='#f0f0f0')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Données MADRS insuffisantes pour l'analyse.")
                
    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Custom tab navigation
    st.markdown("""
    <div class="tab-container">
        <div class="custom-tab active">
            <span style="margin-right: 5px;">📊</span> Distribution des Données
        </div>
        <div class="custom-tab">
            <span style="margin-right: 5px;">📋</span> Patients Récents
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Then use the regular tabs for the actual functionality
    tab1, tab2 = st.tabs([
        "Distribution des Données", 
        "Patients Récents"
    ])
    
    # Distribution des Données tab
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='section-header'>Distribution des Protocoles</div>", unsafe_allow_html=True)
            
            if 'protocol' in st.session_state.final_data.columns:
                # Count patients by protocol
                protocol_counts = st.session_state.final_data['protocol'].value_counts().reset_index()
                protocol_counts.columns = ['Protocole', 'Nombre de Patients']
                
                # Create a pie chart
                fig = px.pie(
                    protocol_counts, 
                    values='Nombre de Patients',
                    names='Protocole',
                    color_discrete_sequence=st.session_state.PASTEL_COLORS
                )
                fig.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#333")
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("La colonne 'protocol' n'existe pas dans les données.")
        
        with col2:
            st.markdown("<div class='section-header'>Distribution des Âges</div>", unsafe_allow_html=True)
            
            if 'age' in st.session_state.final_data.columns:
                fig_age = px.histogram(
                    st.session_state.final_data,
                    x='age',
                    nbins=10,
                    labels={'age': 'Âge', 'count': 'Nombre de Patients'},
                    color_discrete_sequence=[st.session_state.PASTEL_COLORS[2]],
                )
                fig_age.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#333"),
                    xaxis=dict(gridcolor='#f0f0f0'),
                    yaxis=dict(gridcolor='#f0f0f0')
                )
                st.plotly_chart(fig_age, use_container_width=True)
            else:
                st.warning("La colonne 'age' n'existe pas dans les données.")
    
    # Patients Récents tab
    with tab2:
        st.markdown("<div class='section-header'>Patients Récemment Ajoutés</div>", unsafe_allow_html=True)
        
        if 'Timestamp' in st.session_state.final_data.columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_dtype(st.session_state.final_data['Timestamp']):
                st.session_state.final_data['Timestamp'] = pd.to_datetime(
                    st.session_state.final_data['Timestamp'], 
                    errors='coerce'
                )
            
            # Sort by timestamp and get the 5 most recent
            recent_patients = st.session_state.final_data.sort_values(
                'Timestamp', 
                ascending=False
            ).head(5)[['ID', 'Timestamp', 'age', 'protocol']]
            
            if not recent_patients.empty:
                # Format for display
                display_df = recent_patients.copy()
                display_df.columns = ['ID Patient', 'Date d\'ajout', 'Âge', 'Protocole']
                
                # Format date for better display
                display_df['Date d\'ajout'] = display_df['Date d\'ajout'].dt.strftime('%d/%m/%Y')
                
                # Display the table with styling
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("Aucun patient avec horodatage valide.")
        else:
            # Alternative: just show the first 5 patients
            st.info("Pas d'horodatage disponible. Affichage des premiers patients:")
            display_df = st.session_state.final_data[['ID', 'age', 'protocol']].head(5)
            display_df.columns = ['ID Patient', 'Âge', 'Protocole']
            st.dataframe(display_df, use_container_width=True, hide_index=True)