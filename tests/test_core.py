import base64
import json
import urllib.error
from io import BytesIO
from pathlib import Path

import pytest

from openai_image import (
    OpenAIAPIError,
    OpenAIImageCLIError,
    classify_reference,
    create_image,
    normalize_image_request,
    parse_size,
)
from openai_image.core import choose_transport


def test_parse_size_and_gpt_25_validation(tmp_path):
    assert parse_size("1536X1024")["pixels"] == 1_572_864
    request = normalize_image_request(
        {"prompt": "hello", "size": "1536x1024", "quality": "max"}, cwd=tmp_path
    )
    assert request.model == "gpt-image-2.5-flare"
    assert request.size == "1536x1024"

    with pytest.raises(OpenAIImageCLIError, match="divisible by 16"):
        normalize_image_request({"prompt": "hello", "size": "1025x1024"}, cwd=tmp_path)


def test_default_output_dir_is_working_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_IMAGE_OUTPUT_DIR", raising=False)
    request = normalize_image_request({"prompt": "hello"}, cwd=tmp_path)
    assert request.output_dir == tmp_path.resolve()


def test_reference_classification(tmp_path):
    assert classify_reference("https://example.com/a.png", tmp_path).kind == "url"
    assert classify_reference("data:image/png;base64,AAAA", tmp_path).kind == "data_url"
    assert classify_reference("file-Abc_123", tmp_path).kind == "file_id"
    local = classify_reference("@image.png", tmp_path)
    assert local.kind == "local"
    assert local.value == str(tmp_path / "image.png")


def test_transport_selection(tmp_path):
    local = normalize_image_request(
        {"prompt": "edit", "reference_images": ["image.png"]}, cwd=tmp_path
    )
    remote = normalize_image_request(
        {"prompt": "edit", "reference_images": ["https://example.com/image.png"]},
        cwd=tmp_path,
    )
    generated = normalize_image_request({"prompt": "generate"}, cwd=tmp_path)
    assert choose_transport(local) == "multipart"
    assert choose_transport(remote) == "json"
    assert choose_transport(generated) == "json"


def test_dry_run_does_not_need_key_or_existing_reference(tmp_path):
    result = create_image(
        {
            "prompt": "A small owl",
            "reference_images": ["missing.png"],
            "quality": "high",
            "dry_run": True,
            "output_dir": "out",
            "output_prefix": "owl",
        },
        cwd=tmp_path,
    )
    assert result["ok"] is True
    assert result["endpoint"] == "edits"
    assert result["transport"] == "multipart"
    assert result["output_dir"] == str(tmp_path / "out")
    assert result["confirmation_reasons"] == [
        "high quality output",
        "1 reference image(s)",
    ]


class FakeResponse:
    def __init__(self, payload, headers=None):
        self.payload = payload
        self.headers = headers or {}

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_generation_saves_images_and_sanitized_metadata(tmp_path, monkeypatch):
    image = b"not-really-a-png"
    payload = {
        "data": [
            {"b64_json": base64.b64encode(image).decode(), "revised_prompt": "owl"}
        ],
        "output_format": "png",
        "usage": {"total_tokens": 42},
    }
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return FakeResponse(payload, {"x-request-id": "req_test"})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    output = tmp_path / "out"
    (output).mkdir()
    (output / "owl.png").write_bytes(b"existing")

    result = create_image(
        {"prompt": "An owl", "output_dir": str(output), "output_prefix": "owl"},
        api_key="test-key",
    )

    assert len(calls) == 1
    posted = calls[0][0]
    assert posted.full_url.endswith("/images/generations")
    assert posted.get_header("Authorization") == "Bearer test-key"
    assert json.loads(posted.data)["model"] == "gpt-image-2.5-flare"
    assert Path(result["files"][0]["path"]).name == "owl-2.png"
    assert Path(result["files"][0]["path"]).read_bytes() == image
    assert result["request_id"] == "req_test"

    metadata = json.loads(Path(result["metadata_path"]).read_text())
    response_item = metadata["response"]["data"][0]
    assert response_item["has_b64_json"] is True
    assert "b64_json" not in response_item
    assert response_item["revised_prompt"] == "owl"


def test_local_edit_uses_multipart(tmp_path, monkeypatch):
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"reference bytes")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        return FakeResponse(
            {"data": [{"b64_json": base64.b64encode(b"edited").decode()}]},
            {"openai-request-id": "req_edit"},
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = create_image(
        {
            "prompt": "Edit it",
            "reference_images": [str(reference)],
            "output_dir": str(tmp_path / "out"),
            "output_prefix": "edit",
        },
        api_key="key",
    )

    request = captured["request"]
    assert request.full_url.endswith("/images/edits")
    assert request.get_header("Content-type").startswith(
        "multipart/form-data; boundary="
    )
    assert b'name="image[]"' in request.data
    assert b"reference bytes" in request.data
    assert result["transport"] == "multipart"


def test_api_error_preserves_details(tmp_path, monkeypatch):
    body = json.dumps(
        {"error": {"message": "bad request", "code": "bad", "type": "invalid"}}
    ).encode()

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "Bad Request",
            {"x-request-id": "req_bad"},
            BytesIO(body),
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(OpenAIAPIError) as caught:
        create_image({"prompt": "hello", "output_dir": str(tmp_path)}, api_key="key")
    assert caught.value.status == 400
    assert caught.value.request_id == "req_bad"
    assert caught.value.code == "bad"


def test_remote_edit_uses_json_objects(tmp_path, monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        return FakeResponse(
            {"data": [{"b64_json": base64.b64encode(b"edited").decode()}]}
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = create_image(
        {
            "prompt": "Edit from remote references",
            "reference_images": [
                "https://example.com/reference.png",
                "file-Abc_123",
            ],
            "mask": "https://example.com/mask.png",
            "output_dir": str(tmp_path / "out"),
            "output_prefix": "remote-edit",
            "save_metadata": False,
        },
        api_key="key",
    )

    request = captured["request"]
    payload = json.loads(request.data)
    assert request.get_header("Content-type") == "application/json"
    assert payload["images"] == [
        {"image_url": "https://example.com/reference.png"},
        {"file_id": "file-Abc_123"},
    ]
    assert payload["mask"] == {"image_url": "https://example.com/mask.png"}
    assert result["transport"] == "json"
    assert "metadata_path" not in result
    assert not (tmp_path / "out" / "remote-edit.json").exists()


@pytest.mark.parametrize(
    ("request_data", "message"),
    [
        ({"n": 0}, "n must be between 1 and 10"),
        ({"output_compression": 50}, "only supported for jpeg or webp"),
        (
            {"output_format": "jpeg", "background": "transparent"},
            "transparent backgrounds require png or webp",
        ),
        ({"mask": "mask.png"}, "mask requires at least one reference image"),
        (
            {"model": "gpt-image-1", "quality": "max"},
            "quality xhigh and max require",
        ),
    ],
)
def test_request_validation_errors(tmp_path, request_data, message):
    with pytest.raises(OpenAIImageCLIError, match=message):
        normalize_image_request({"prompt": "test", **request_data}, cwd=tmp_path)
