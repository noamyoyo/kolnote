from kolnote.config import apply_env, load_settings


def test_env_builds_config_without_file():
    s = load_settings(
        env={
            "KOLNOTE_STT__TYPE": "faster_whisper",
            "KOLNOTE_STT__MODEL": "large-v3",
            "KOLNOTE_STT__BEAM_SIZE": "3",
            "KOLNOTE_STT__VAD_FILTER": "false",
            "KOLNOTE_POLICY__ALLOW_CHATS": "a@g.us, b@g.us",
            "KOLNOTE_LANGUAGE": "he",
            "UNRELATED": "ignored",
        }
    )
    assert s.stt == {"type": "faster_whisper", "model": "large-v3", "beam_size": 3, "vad_filter": False}
    assert s.policy == {"allow_chats": ["a@g.us", "b@g.us"]}
    assert s.language == "he"


def test_env_overrides_file_and_keeps_string_type(tmp_path):
    cfg = tmp_path / "c.toml"
    cfg.write_text('[stt]\ntype = "faster_whisper"\nmodel = "small"\ninitial_prompt = "x"\n')
    s = load_settings(cfg, env={"KOLNOTE_STT__MODEL": "large-v3", "KOLNOTE_STT__INITIAL_PROMPT": "123"})
    assert s.stt["type"] == "faster_whisper"
    assert s.stt["model"] == "large-v3"
    assert s.stt["initial_prompt"] == "123"


def test_json_values_and_no_mutation_of_other_sections():
    data = apply_env({"channel": {"type": "folder"}}, {"KOLNOTE_STT__EXTRA": '["a", "b"]'})
    assert data["stt"]["extra"] == ["a", "b"]
    assert data["channel"] == {"type": "folder"}
