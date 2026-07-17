"""
test_explain_smoke.py

Smoke test for explain._get_api_key()'s lookup order: OS environment
variables first, then Streamlit's st.secrets (if streamlit happens to be
importable), and a safe None if neither has anything -- all without ever
raising, since streamlit isn't installed in every environment this project
runs in (e.g. this test suite's sandbox).

Run: python test_explain_smoke.py
"""

import os
import sys
import types

import explain


def _clear_env(*names):
    for name in names:
        os.environ.pop(name, None)


def test_no_key_anywhere_returns_none():
    _clear_env("ANTHROPIC_API_KEY")
    assert explain._get_api_key("ANTHROPIC_API_KEY") is None
    print("[ok] no env var, no streamlit installed/secrets configured -> None")


def test_env_var_is_found():
    _clear_env("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-from-env"
    try:
        assert explain._get_api_key("ANTHROPIC_API_KEY") == "sk-ant-from-env"
        print("[ok] OS environment variable is picked up")
    finally:
        _clear_env("ANTHROPIC_API_KEY")


def test_env_var_takes_priority_over_secrets():
    """If both an env var AND st.secrets have a value, env var wins (checked
    first) -- matches the documented order in explain.py's module docstring."""
    _clear_env("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-from-env"

    fake_st = types.SimpleNamespace(secrets={"ANTHROPIC_API_KEY": "sk-ant-from-secrets"})
    sys.modules["streamlit"] = fake_st
    try:
        assert explain._get_api_key("ANTHROPIC_API_KEY") == "sk-ant-from-env"
        print("[ok] env var takes priority over st.secrets when both are set")
    finally:
        _clear_env("ANTHROPIC_API_KEY")
        sys.modules.pop("streamlit", None)


def test_falls_back_to_secrets_when_no_env_var():
    _clear_env("ANTHROPIC_API_KEY")
    fake_st = types.SimpleNamespace(secrets={"ANTHROPIC_API_KEY": "sk-ant-from-secrets"})
    sys.modules["streamlit"] = fake_st
    try:
        assert explain._get_api_key("ANTHROPIC_API_KEY") == "sk-ant-from-secrets"
        print("[ok] falls back to st.secrets when no env var is set")
    finally:
        sys.modules.pop("streamlit", None)


def test_missing_streamlit_module_is_safe():
    """No streamlit installed at all (import raises) -- must not crash, just
    behave as if no secrets are configured."""
    _clear_env("ANTHROPIC_API_KEY")
    sys.modules.pop("streamlit", None)  # ensure a real import attempt happens
    assert explain._get_api_key("ANTHROPIC_API_KEY") is None
    print("[ok] streamlit not installed -> _get_api_key degrades safely to None, no crash")


def test_second_name_checked_if_first_missing():
    """Gemini's dual env var name (GOOGLE_API_KEY / GEMINI_API_KEY) style call."""
    _clear_env("GOOGLE_API_KEY", "GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = "gm-key"
    try:
        assert explain._get_api_key("GOOGLE_API_KEY", "GEMINI_API_KEY") == "gm-key"
        print("[ok] second name is checked when the first isn't set (env var case)")
    finally:
        _clear_env("GOOGLE_API_KEY", "GEMINI_API_KEY")


def main():
    test_no_key_anywhere_returns_none()
    test_env_var_is_found()
    test_env_var_takes_priority_over_secrets()
    test_falls_back_to_secrets_when_no_env_var()
    test_missing_streamlit_module_is_safe()
    test_second_name_checked_if_first_missing()
    print("\nAll explain.py smoke-test assertions passed.")


if __name__ == "__main__":
    main()
