# components/sidebar.py
import streamlit as st
import re
import logging
from datetime import datetime

# --- Helper Function ---
def extract_number(id_str):
    """Extract numeric part from a patient ID for sorting"""
    match = re.search(r'\d+', str(id_str)) # Ensure input is string
    return int(match.group()) if match else float('inf') # Return infinity for non-numeric to sort last

# --- Role-Based Page Access ---
# Define role permissions
ROLE_PERMISSIONS = {
    "admin": [ "Vue d'Ensemble", "Tableau de Bord du Patient", "Parcours Patient", "Analyse des Protocoles",
               "Plan de Soins et Entrées Infirmières", "Suivi des Effets Secondaires", "Performance ML", "Déconnexion"],
    "md": [ "Vue d'Ensemble", "Tableau de Bord du Patient", "Parcours Patient", "Analyse des Protocoles",
             "Suivi des Effets Secondaires", "Performance ML", "Déconnexion"],
    "nurse": [ "Vue d'Ensemble", "Tableau de Bord du Patient", "Parcours Patient",
               "Plan de Soins et Entrées Infirmières", "Suivi des Effets Secondaires", "Déconnexion"],
    "default": ["Vue d'Ensemble", "Déconnexion"] # Fallback role
}

# --- Main Sidebar Rendering Function ---
def render_sidebar():
    """Render the sidebar with role-based navigation and patient selection"""
    with st.sidebar:
        st.title("🧠 Tableau de Bord TMS")

        # Role display moved to app.py after login check

        st.markdown("---")

        # --- PATIENT SELECTION SECTION ---
        st.markdown("### 👥 Sélection Patient")
        patient_list = [] # Initialize
        selected_index = 0 # Initialize

        if 'final_data' not in st.session_state or st.session_state.final_data.empty:
            st.warning("Données patient non chargées.")
        else:
            try:
                # Get unique, non-null patient IDs
                if 'ID' in st.session_state.final_data.columns:
                     existing_patient_ids = st.session_state.final_data['ID'].dropna().unique().tolist()
                     # Ensure IDs are strings before sorting, handle potential non-string IDs gracefully
                     patient_list = sorted([str(pid) for pid in existing_patient_ids if pid is not None], key=extract_number)
                else:
                     st.error("Colonne 'ID' manquante dans les données patient.")
                     patient_list = [] # Ensure patient_list is empty list on error


                # Determine selected index
                current_selection = st.session_state.get('selected_patient_id', None)
                if current_selection is not None and str(current_selection) in patient_list: # Compare as string
                    selected_index = patient_list.index(str(current_selection))
                elif patient_list: # If current selection invalid or None, default to first patient
                    selected_index = 0
                    st.session_state.selected_patient_id = patient_list[0] # Update state only if list not empty
                else: # No patients available or ID column missing
                     st.session_state.selected_patient_id = None

            except Exception as e:
                 st.error(f"Erreur préparation liste patients: {e}")
                 logging.exception("Error preparing patient list")
                 patient_list = []
                 selected_index = 0
                 st.session_state.selected_patient_id = None

        # Patient selection dropdown
        if patient_list:
            selected_patient = st.selectbox(
                "👤 Sélectionner un Patient:", patient_list, index=selected_index,
                help="Choisissez un patient pour voir ses données détaillées.",
                 key="sidebar_patient_selector" # Use consistent key
            )
            # Update session state if selection changed
            # Compare potentially numeric session state with string from selectbox
            if str(selected_patient) != str(st.session_state.get('selected_patient_id')):
                 st.session_state.selected_patient_id = selected_patient # Store selection as is (likely string)
                 st.rerun() # Rerun to reflect change immediately
            # Simple display of current selection below dropdown
            st.write(f"Patient actuel: **{st.session_state.selected_patient_id}**")

        else:
             st.error("Aucun patient disponible.")


        st.markdown("---")

        # --- NAVIGATION SECTION (ROLE-BASED) ---
        st.markdown("### 📋 Navigation")

        # Define ALL possible navigation options (without PID-5)
        all_main_options = {
            "Vue d'Ensemble": "Statistiques générales de la cohorte.",
            "Tableau de Bord du Patient": "Vue détaillée du patient (évaluations, plan, etc.).",
            "Parcours Patient": "Chronologie des événements clés du patient.",
            "Analyse des Protocoles": "Comparaison de l'efficacité des protocoles TMS.",
            "Plan de Soins et Entrées Infirmières": "Ajouter/modifier le plan de soins et historique.",
            "Suivi des Effets Secondaires": "Ajouter/voir l'historique des effets secondaires.",
            "Performance ML": "Analyse des performances des modèles d'apprentissage automatique.",
            "Déconnexion": "Se déconnecter du système."
        }

        # Filter options based on user role
        user_role = st.session_state.get('role', 'default')
        allowed_pages = ROLE_PERMISSIONS.get(user_role, ROLE_PERMISSIONS["default"])
        available_options = {page: desc for page, desc in all_main_options.items() if page in allowed_pages}

        # Store allowed pages in session state
        st.session_state.allowed_pages = list(available_options.keys())

        # Get current selection, ensure it's valid for the role
        current_page = st.session_state.get('sidebar_selection', None)
        if current_page not in available_options:
             current_page = list(available_options.keys())[0] if available_options else None
             st.session_state.sidebar_selection = current_page # Update state if defaulted

        if not available_options:
             st.warning("Aucune page disponible pour votre rôle.")
             return None # Return None if no pages are available

        # Display radio buttons ONLY for allowed pages
        selected_option = st.radio(
            "Sélectionner une page:",
            options=list(available_options.keys()),
            index=list(available_options.keys()).index(current_page) if current_page in available_options else 0,
             key="sidebar_navigation"
        )

        # Display help text for the selected option
        st.info(f"**Page Actuelle:** {selected_option}\n\n*{available_options[selected_option]}*")

        # Update session state if selection changed and rerun
        if 'sidebar_selection' not in st.session_state or selected_option != st.session_state.sidebar_selection:
            st.session_state.sidebar_selection = selected_option
            st.rerun()

        st.markdown("---")
        # --- Help and Stats Sections ---
        with st.expander("❓ Aide"):
             st.markdown("Naviguez entre vos vues disponibles à l'aide des options ci-dessus. Sélectionnez un patient pour voir ses détails spécifiques.")
        if 'session_started' in st.session_state:
            with st.expander("⏱️ Session Info"):
                # This calculation should now work
                session_duration = datetime.now() - st.session_state.session_started
                st.write(f"Durée Session: {str(session_duration).split('.')[0]}")
                view_count = len(st.session_state.get('patient_views', {}))
                st.write(f"Consultations Patient: {view_count}")


    # Return the page selected by the user
    return st.session_state.get('sidebar_selection')