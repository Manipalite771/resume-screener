# Resume Screener

Streamlit app for screening PDF resumes for GenAI Delivery Lead, Lead Business
Analyst, and Agentforce Engineer roles.

## Shared sign-in

The app requires a shared user ID and password before showing any screening
controls. Give those credentials and the app URL to approved users.

For local use, copy `.streamlit/secrets.toml.example` to
`.streamlit/secrets.toml` and set `APP_USERNAME`, `APP_PASSWORD`, and
`AZURE_OPENAI_KEY`. If the file already exists, keep the existing Azure key and
add the two sign-in settings. Use a long, unique password. Never commit the
real secrets file; it is ignored by Git.

For Streamlit Community Cloud, add the same settings in the app's **Settings →
Secrets** before deploying this change. Local secrets are not uploaded by Git.
If the sign-in settings are missing or empty, the app stays locked. The sign-in
settings may also be supplied as server environment variables, which take
precedence over Streamlit secrets.

Users can sign out from the sidebar. Sessions expire after 12 hours and may
require sign-in again after a refresh or connection loss. Signing out clears
the session, including resume widget state and a manually entered API key.
Changing either shared credential invalidates existing sessions on their next
interaction. Rotate the password to withdraw previously shared access.

Five incorrect attempts pause sign-in for one minute within that browser session.
This is basic shared-account access: it does not provide individual user
revocation, per-user audit records, or distributed rate limiting.

## Run locally

```sh
python -m pip install -r requirements.txt
streamlit run app.py
```

## Verify

```sh
python -m unittest discover -s tests -v
```
