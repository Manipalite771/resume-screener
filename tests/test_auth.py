import os
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.proto.TextInput_pb2 import TextInput
from streamlit.testing.v1 import AppTest

import screening_logic


APP = Path(__file__).resolve().parents[1] / "app.py"
TEST_USERNAME = "test-approved-user"
TEST_PASSWORD = "Only-for-tests-42!"


class SignInTests(unittest.TestCase):
    def setUp(self):
        # Keep tests independent of real local or hosting credentials.
        environment = {key: value for key, value in os.environ.items()
                       if key not in {"APP_USERNAME", "APP_PASSWORD"}}
        self.env_patch = patch.dict(os.environ, environment, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.request_patch = patch("requests.post")
        self.requests = self.request_patch.start()
        self.addCleanup(self.request_patch.stop)

    def app(self, **overrides):
        app = AppTest.from_file(APP, default_timeout=10)
        app.secrets = {
            "APP_USERNAME": TEST_USERNAME,
            "APP_PASSWORD": TEST_PASSWORD,
            "AZURE_OPENAI_KEY": "test-key-no-network",
            **overrides,
        }
        return app.run()

    def sign_in(self, app, username=TEST_USERNAME, password=TEST_PASSWORD):
        app.text_input(key="_login_user").set_value(username)
        app.text_input(key="_login_password").set_value(password)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)

    def assert_locked(self, app):
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.selectbox), 0)
        self.assertEqual(len(app.get("file_uploader")), 0)
        self.requests.assert_not_called()

    def test_logged_out_cannot_access_screening(self):
        app = self.app()
        app.query_params["authenticated"] = "true"
        app.run()
        self.assert_locked(app)
        self.assertEqual([field.label for field in app.text_input], ["User ID", "Password"])
        self.assertEqual(app.text_input[1].proto.type, TextInput.PASSWORD)

    def test_app_refresh_reloads_deployed_screening_rules(self):
        analysis = {
            "experience": {"total_professional_years": 3, "evidence_status": "demonstrated"},
            "criteria": {
                key: {"score": maximum, "evidence_status": "demonstrated"}
                for key, (_, maximum) in screening_logic.GENAI_CRITERIA.items()
            },
        }
        # Reproduce the live worker retaining the pre-deployment six-year threshold.
        with patch.object(screening_logic, "GENAI_MINIMUM_YEARS", 6):
            self.assertFalse(screening_logic.evaluate_genai_analysis(analysis, "PASS")["proceed"])
            self.assert_locked(self.app())
            decision = screening_logic.evaluate_genai_analysis(analysis, "PASS")
            self.assertTrue(decision["gates"]["minimum_experience"])
            self.assertTrue(decision["proceed"])

    def test_missing_or_invalid_configuration_keeps_app_locked(self):
        for setting in ("APP_USERNAME", "APP_PASSWORD"):
            for value in (None, "", "   ", 123):
                with self.subTest(setting=setting, value=value):
                    app = self.app(**{setting: value})
                    self.assert_locked(app)
                    self.assertIn("not configured", app.error[0].value)
                    self.assertEqual(len(app.button), 0)

    def test_wrong_credentials_are_rejected_without_retaining_password(self):
        for username, password in ((TEST_USERNAME, "wrong"), ("wrong", TEST_PASSWORD), ("", "")):
            with self.subTest(username=username):
                app = self.app()
                self.sign_in(app, username, password)
                self.assert_locked(app)
                self.assertEqual(app.error[0].value, "Incorrect user ID or password.")
                self.assertFalse(app.session_state.filtered_state.get("_login_password"))

    def test_login_persists_through_role_changes_and_logout_clears_session(self):
        app = self.app()
        self.sign_in(app)
        self.assertEqual(len(app.selectbox), 1)
        self.assertEqual(len(app.get("file_uploader")), 1)
        self.assertEqual(len(app.text_input), 0)
        self.assertNotIn(TEST_PASSWORD, repr(app.session_state.filtered_state))
        app.selectbox[0].select("Agentforce Engineer").run()
        self.assertEqual(app.selectbox[0].value, "Agentforce Engineer")
        app.session_state["openai_api_key"] = "temporary-user-key"
        app.session_state["resume_result"] = "private-candidate-data"
        app.sidebar.button[0].click().run()
        self.assert_locked(app)
        state = app.session_state.filtered_state
        self.assertNotIn("openai_api_key", state)
        self.assertNotIn("resume_result", state)
        self.assertNotIn("_auth_fingerprint", state)
        app.run()
        self.assert_locked(app)

    def test_new_browser_session_requires_its_own_login(self):
        first = self.app()
        self.sign_in(first)
        self.assert_locked(self.app())

    def test_expired_session_is_locked_and_cleared(self):
        app = self.app()
        self.sign_in(app)
        app.session_state["_auth_expires_at"] = 0
        app.session_state["resume_result"] = "private-candidate-data"
        app.run()
        self.assert_locked(app)
        self.assertNotIn("resume_result", app.session_state.filtered_state)

    def test_changing_either_credential_revokes_existing_session(self):
        for setting in ("APP_USERNAME", "APP_PASSWORD"):
            with self.subTest(setting=setting):
                app = self.app()
                self.sign_in(app)
                app.secrets[setting] = "rotated-value"
                app.run()
                self.assert_locked(app)
                self.sign_in(app)
                self.assert_locked(app)

    def test_unicode_credentials_work(self):
        app = self.app(APP_USERNAME="採用担当", APP_PASSWORD="共有パスワード-42!")
        self.sign_in(app, "採用担当", "共有パスワード-42!")
        self.assertEqual(len(app.selectbox), 1)

    def test_session_cooldown_blocks_login_until_it_expires(self):
        app = self.app()
        for _ in range(5):
            self.sign_in(app, password="wrong")
        self.assertIn("Too many attempts", app.error[0].value)
        self.sign_in(app)
        self.assert_locked(app)
        app.session_state["_login_retry_at"] = 0
        self.sign_in(app)
        self.assertEqual(len(app.selectbox), 1)

    def test_environment_credentials_override_secrets(self):
        with patch.dict(os.environ, {"APP_USERNAME": "env-user", "APP_PASSWORD": "env-password"}):
            app = self.app()
            self.sign_in(app)
            self.assert_locked(app)
            self.sign_in(app, "env-user", "env-password")
            self.assertEqual(len(app.selectbox), 1)


if __name__ == "__main__":
    unittest.main()
