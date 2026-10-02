import asyncio

from tests.test_core import FakeSTT
from kolnote import bench, pipeline
from kolnote.adapters.channels.folder import FolderChannel
from kolnote.policy import Policy


def test_folder_channel_writes_transcript_next_to_audio(tmp_path):
    (tmp_path / "a.ogg").write_bytes(b"x")
    (tmp_path / "done.ogg").write_bytes(b"x")
    (tmp_path / "done.txt").write_text("already")
    asyncio.run(pipeline.run(FolderChannel(tmp_path, once=True), FakeSTT(text="hi"), Policy(open=True)))
    assert (tmp_path / "a.txt").read_text().strip() == "hi"
    assert (tmp_path / "done.txt").read_text() == "already"


def test_bench_scores_dataset(tmp_path):
    (tmp_path / "one.ogg").write_bytes(b"x")
    (tmp_path / "one.txt").write_text("a b c d")
    (tmp_path / "unlabeled.ogg").write_bytes(b"x")
    result = asyncio.run(bench.run_bench(tmp_path, FakeSTT(text="a b c e"), language="he"))
    assert len(result.files) == 1
    assert result.wer == 0.25
    assert "WER" in bench.format_summary([("fake", result)])
