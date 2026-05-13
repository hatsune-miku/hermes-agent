from __future__ import annotations


class DummyConsole:
    def print(self, *_args, **_kwargs):
        pass


def test_setup_password_prompt_uses_visible_input(monkeypatch):
    from hermes_cli import setup

    monkeypatch.setattr("builtins.input", lambda _prompt="": "visible-secret")
    monkeypatch.setattr(
        "getpass.getpass",
        lambda _prompt="": (_ for _ in ()).throw(AssertionError("getpass should not be used")),
    )

    assert setup.prompt("API key", password=True) == "visible-secret"


def test_cli_output_password_prompt_uses_visible_input(monkeypatch):
    from hermes_cli import cli_output

    monkeypatch.setattr("builtins.input", lambda _prompt="": "visible-secret")
    monkeypatch.setattr(
        "getpass.getpass",
        lambda _prompt="": (_ for _ in ()).throw(AssertionError("getpass should not be used")),
    )

    assert cli_output.prompt("API key", password=True) == "visible-secret"


def test_plugin_secret_prompt_uses_visible_input(monkeypatch):
    from hermes_cli.plugins_cmd import _prompt_plugin_env_vars

    manifest = {
        "requires_env": [
            {
                "name": "PLUGIN_API_KEY",
                "description": "Plugin API key",
                "secret": True,
            }
        ]
    }
    saved = {}

    monkeypatch.setattr("builtins.input", lambda _prompt="": "visible-secret")
    monkeypatch.setattr(
        "getpass.getpass",
        lambda _prompt="": (_ for _ in ()).throw(AssertionError("getpass should not be used")),
    )
    monkeypatch.setattr("hermes_cli.config.save_env_value", lambda key, value: saved.__setitem__(key, value))
    monkeypatch.setattr("hermes_cli.config.get_env_value", lambda _key: None)

    _prompt_plugin_env_vars(manifest, console=DummyConsole())

    assert saved == {"PLUGIN_API_KEY": "visible-secret"}
