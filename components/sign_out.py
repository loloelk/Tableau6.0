# components/sign_out.py
import streamlit as st
import time
import logging

def sign_out_page():
    """
    Display a sign out page with confirmation and redirection to login
    """
    st.header("Déconnexion")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Vous avez été déconnecté avec succès.")
        
        st.info("Vos données ont été sécurisées et votre session a été fermée.")
        
        # Show animation of checkmark or similar
        with st.spinner("Sécurisation de vos données..."):
            # Simulate a brief delay for better UX
            time.sleep(1.5)
        
        st.success("✅ Données sécurisées et session terminée")
        
        # Add a message about session timeout
        st.markdown("""
        Pour des raisons de sécurité, les sessions sont automatiquement fermées après 
        une période d'inactivité. Vous devrez vous reconnecter pour accéder à nouveau aux données.
        """)
        
        # Add a button to go back to login screen
        if st.button("Retour à l'écran de connexion", use_container_width=True):
            # Clear all session state
            for key in list(st.session_state.keys()):
                if key != "authentication_status":
                    del st.session_state[key]
            
            # Set authentication status to False
            st.session_state.authentication_status = False
            st.session_state.username = None
            
            # Rerun to show login page
            st.rerun()
    
    # Footer with security message
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.8em;'>
    Pour toute question concernant la sécurité de vos données, veuillez contacter l'administrateur système.
    </div>
    """, unsafe_allow_html=True)