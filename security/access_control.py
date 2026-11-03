"""Shared password gate for the private Memory Vault demo."""
import hmac

import streamlit as st

from config.settings import settings


def require_private_access() -> None:
    """Fail closed until the deployment password is configured and entered."""
    expected = settings.APP_ACCESS_PASSWORD
    if not expected:
        st.error(
            "Private access is not configured. Add APP_ACCESS_PASSWORD "
            "to your environment or Streamlit secrets."
        )
        st.stop()

    if st.session_state.get("_private_access_granted") is True:
        return

    with st.form("private_access_form"):
        supplied = st.text_input("Access password", type="password")
        submitted = st.form_submit_button("Unlock")

    if submitted:
        if hmac.compare_digest(supplied, expected):
            st.session_state["_private_access_granted"] = True
            st.rerun()
        st.error("Incorrect password.")

    st.stop()
