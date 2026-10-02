import json

from openai_image.cli import main


def test_cli_dry_run_json(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = main(
        [
            "--prompt",
            "Tiny owl",
            "--dry-run",
            "--output-dir",
            "results",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    result = json.loads(captured.out)
    assert result["dry_run"] is True
    assert result["output_dir"] == str(tmp_path / "results")


def test_cli_json_input_is_overridden_and_references_are_appended(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "prompt": "from json",
                "quality": "low",
                "reference_images": ["first.png"],
                "dry_run": True,
            }
        )
    )
    code = main(
        [
            "--input",
            str(request_path),
            "--prompt",
            "from cli",
            "--quality",
            "medium",
            "--reference",
            "second.png",
            "--json",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["prompt"] == "from cli"
    assert result["quality"] == "medium"
    assert len(result["reference_images"]) == 2


def test_cli_validation_error_is_json(capsys):
    code = main(["--prompt", "hello", "--n", "99", "--dry-run", "--json"])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["ok"] is False
    assert error["message"] == "n must be between 1 and 10."


def test_cli_missing_prompt_returns_usage_error(capsys):
    code = main(["--dry-run"])
    captured = capsys.readouterr()
    assert code == 2
    assert "usage:" in captured.err
    assert "prompt is required" in captured.err
