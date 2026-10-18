from src.config import load_config


def test_a_config_can_extend_another_and_override_only_some_settings(tmp_path):
    (tmp_path / "base.yaml").write_text("group_id: G1\napp:\n  name: A\n  theme: dark\n", encoding="utf-8")
    (tmp_path / "second.yaml").write_text("extends: base.yaml\napp:\n  name: B\n", encoding="utf-8")
    assert load_config(tmp_path / "second.yaml") == {"group_id": "G1", "app": {"name": "B", "theme": "dark"}}
