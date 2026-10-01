"""Shared-account sign-in for the Streamlit app."""

import hashlib
import hmac
import os
import time

import streamlit as st


SESSION_SECONDS = 12 * 60 * 60
MAX_ATTEMPTS = 5
RETRY_SECONDS = 60


def _setting(name):
    """Read private server configuration; never use a public default."""
    if name in os.environ:
        value = os.environ[name]
    else:
        try:
            value = st.secrets.get(name, "")
        except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
            value = ""
    return value if isinstance(value, str) else ""


def _fingerprint(username, password):
    # Store a configuration fingerprint, never the password, in session state.
    return hashlib.sha256(repr((username, password)).encode("utf-8")).hexdigest()


def _clear_session():
    # Remove uploaded resume widget state and any manually entered API key too.
    for key in list(st.session_state):
        del st.session_state[key]


def _sign_in(username, password):
    entered_user = st.session_state.pop("_login_user", "")
    entered_password = st.session_state.pop("_login_password", "")
    now = time.time()
    if now < st.session_state.get("_login_retry_at", 0):
        return

    user_matches = hmac.compare_digest(entered_user.encode("utf-8"), username.encode("utf-8"))
    password_matches = hmac.compare_digest(
        entered_password.encode("utf-8"), password.encode("utf-8")
    )
    if user_matches and password_matches:
        _clear_session()
        st.session_state["_auth_fingerprint"] = _fingerprint(username, password)
        st.session_state["_auth_expires_at"] = now + SESSION_SECONDS
        return

    attempts = st.session_state.get("_login_attempts", 0) + 1
    st.session_state["_login_error"] = "Incorrect user ID or password."
    st.session_state["_login_attempts"] = attempts
    if attempts >= MAX_ATTEMPTS:
        st.session_state["_login_retry_at"] = now + RETRY_SECONDS
        st.session_state["_login_attempts"] = 0


def require_login():
    """Stop every unauthenticated run before any screening UI or API access."""
    username = _setting("APP_USERNAME")
    password = _setting("APP_PASSWORD")
    if not username.strip() or not password.strip():
        _clear_session()
        st.title("Resume Screener")
        st.error("Sign-in is not configured. Please contact the app owner.")
        st.stop()

    now = time.time()
    fingerprint = _fingerprint(username, password)
    if (
        st.session_state.get("_auth_fingerprint") == fingerprint
        and now < st.session_state.get("_auth_expires_at", 0)
    ):
        with st.sidebar:
            st.caption(f"Signed in as {username}")
            st.button("Sign out", on_click=_clear_session)
        return

    if "_auth_fingerprint" in st.session_state:
        _clear_session()

    _, login_column, _ = st.columns([1, 2, 1])
    with login_column:
        st.title("Resume Screener")
        st.markdown("Sign in to screen resumes.")
        st.caption("Use the user ID and password shared with you by the app owner.")
        with st.form("sign_in", clear_on_submit=True):
            st.text_input("User ID", key="_login_user")
            st.text_input("Password", type="password", key="_login_password")
            st.form_submit_button(
                "Sign in",
                type="primary",
                use_container_width=True,
                on_click=_sign_in,
                args=(username, password),
            )

        retry_at = st.session_state.get("_login_retry_at", 0)
        if now < retry_at:
            seconds = max(1, int(retry_at - now) + 1)
            st.error(f"Too many attempts. Please try again in {seconds} seconds.")
        elif st.session_state.get("_login_error"):
            st.error(st.session_state["_login_error"])

    st.stop()
