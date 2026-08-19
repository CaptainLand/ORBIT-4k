from pathlib import Path

from orbit4k.preprocessing.osu_parser import parse_osu_file


def test_non_4k_is_rejected(tmp_path: Path):
    text = """osu file format v14
[General]
AudioFilename: audio.mp3
Mode: 3
[Difficulty]
CircleSize:7
[TimingPoints]
0,500,4,2,1,50,1,0
[HitObjects]
64,192,0,1,0,0:0:0:0:
"""
    (tmp_path / "audio.mp3").write_bytes(b"x")
    path = tmp_path / "map.osu"
    path.write_text(text)
    try:
        parse_osu_file(path, calculate_stars=False)
        assert False, "7K should be rejected"
    except ValueError as exc:
        assert "not osu!mania 4K" in str(exc)
