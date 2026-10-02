"""Command-line interface for openai-image."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .core import (
    OpenAIImageCLIError,
    create_image,
    format_result_summary,
)


class CLIArgumentError(OpenAIImageCLIError):
    pass


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CLIArgumentError(message)


class PromptFileAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        try:
            text = Path(str(values)).read_text(encoding="utf-8")
        except OSError as exc:
            raise CLIArgumentError(
                f"Failed to read prompt file {values}: {exc}"
            ) from exc
        setattr(namespace, self.dest, text)


def _version() -> str:
    try:
        return version("openai-image")
    except PackageNotFoundError:
        return "0.1.0"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="openai-image",
        description="Generate or edit images with the OpenAI image API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Environment:
  OPENAI_API_KEY              Required unless --dry-run is used
  OPENAI_BASE_URL             API base URL (default: https://api.openai.com/v1)
  OPENAI_ORGANIZATION         Optional organization header
  OPENAI_PROJECT              Optional project header
  OPENAI_IMAGE_OUTPUT_DIR     Default output directory (default: ./openai-images)
  OPENAI_IMAGE_TIMEOUT        HTTP timeout in seconds (default: 600)

Reference images may be local paths, HTTP(S) URLs, image data URLs, or OpenAI file IDs.
""",
    )
    parser.add_argument(
        "-p", "--prompt", default=argparse.SUPPRESS, help="desired final image"
    )
    parser.add_argument(
        "--prompt-file",
        action=PromptFileAction,
        dest="prompt",
        default=argparse.SUPPRESS,
        metavar="PATH",
        help="read the prompt from a UTF-8 text file",
    )
    parser.add_argument(
        "-r",
        "--reference",
        "--ref",
        action="append",
        dest="reference_images",
        default=argparse.SUPPRESS,
        metavar="PATH_OR_URL",
        help="reference image; repeat up to 16 times",
    )
    parser.add_argument(
        "--mask", default=argparse.SUPPRESS, metavar="PATH_OR_URL", help="edit mask"
    )
    parser.add_argument(
        "-n",
        "--n",
        default=argparse.SUPPRESS,
        metavar="COUNT",
        help="number of images, 1-10",
    )
    parser.add_argument(
        "--size",
        default=argparse.SUPPRESS,
        metavar="WIDTHxHEIGHT",
        help='output size or "auto"',
    )
    parser.add_argument(
        "--quality",
        default=argparse.SUPPRESS,
        metavar="VALUE",
        help="low, medium, high, xhigh, max, or auto",
    )
    parser.add_argument(
        "--format",
        "--output-format",
        dest="output_format",
        default=argparse.SUPPRESS,
        metavar="VALUE",
        help="png, jpeg, or webp",
    )
    parser.add_argument(
        "--compression",
        "--output-compression",
        dest="output_compression",
        default=argparse.SUPPRESS,
        metavar="N",
        help="JPEG/WebP compression, 0-100",
    )
    parser.add_argument(
        "--background",
        default=argparse.SUPPRESS,
        metavar="VALUE",
        help="auto, opaque, or transparent",
    )
    parser.add_argument(
        "--moderation", default=argparse.SUPPRESS, metavar="VALUE", help="auto or low"
    )
    parser.add_argument(
        "--model", default=argparse.SUPPRESS, metavar="MODEL", help="image model"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=argparse.SUPPRESS,
        metavar="PATH",
        help="output directory",
    )
    parser.add_argument(
        "--output-prefix",
        default=argparse.SUPPRESS,
        metavar="NAME",
        help="filename prefix",
    )
    parser.add_argument(
        "--transport",
        default=argparse.SUPPRESS,
        metavar="VALUE",
        help="auto, json, or multipart",
    )
    parser.add_argument(
        "--user", default=argparse.SUPPRESS, metavar="ID", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--input",
        default=argparse.SUPPRESS,
        metavar="PATH",
        help='read a JSON request from a file, or "-" for stdin',
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=argparse.SUPPRESS,
        help="validate without calling OpenAI",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_false",
        dest="save_metadata",
        default=argparse.SUPPRESS,
        help="do not write a metadata sidecar",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="print machine-readable JSON",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    return parser


def _read_input(path: str) -> dict[str, Any]:
    try:
        text = (
            sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
        )
    except OSError as exc:
        raise OpenAIImageCLIError(f"Failed to read JSON input {path}: {exc}") from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OpenAIImageCLIError(f"Failed to parse JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OpenAIImageCLIError("JSON input must contain an object.")
    return value


def parse_request(
    parser: ArgumentParser, argv: Sequence[str]
) -> tuple[dict[str, Any], bool]:
    namespace = parser.parse_args(argv)
    values = vars(namespace)
    json_output = bool(values.pop("json_output", False))
    input_path = values.pop("input", None)
    base = _read_input(input_path) if input_path else {}

    references: list[Any] = []
    for key in ("reference_images", "references"):
        value = base.pop(key, None)
        if isinstance(value, list):
            references.extend(value)
        elif value is not None:
            references.append(value)
    cli_references = values.pop("reference_images", [])
    references.extend(cli_references)

    request = {**base, **values}
    if references:
        request["reference_images"] = references
    return request, json_output


def _error_payload(error: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "name": error.__class__.__name__,
        "message": str(error),
    }
    details = getattr(error, "details", None)
    if details:
        payload["details"] = details
    for attribute, key in (
        ("status", "status"),
        ("request_id", "request_id"),
        ("code", "code"),
    ):
        value = getattr(error, attribute, None)
        if value is not None:
            payload[key] = value
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    json_requested = "--json" in arguments
    try:
        request, json_output = parse_request(parser, arguments)
        json_requested = json_output
        if not request.get("prompt"):
            if json_output:
                print(
                    json.dumps(
                        _error_payload(OpenAIImageCLIError("prompt is required.")),
                        indent=2,
                    ),
                    file=sys.stderr,
                )
            else:
                parser.print_usage(sys.stderr)
                print("openai-image: error: prompt is required.", file=sys.stderr)
            return 2
        result = create_image(request, cwd=Path.cwd())
        print(
            json.dumps(result, indent=2)
            if json_output
            else format_result_summary(result)
        )
        return 0
    except KeyboardInterrupt:
        print("openai-image: interrupted", file=sys.stderr)
        return 130
    except (OpenAIImageCLIError, OSError, ValueError) as error:
        payload = _error_payload(error)
        if json_requested:
            print(json.dumps(payload, indent=2), file=sys.stderr)
        else:
            print(f"{payload['name']}: {payload['message']}", file=sys.stderr)
            if payload.get("request_id"):
                print(f"OpenAI request id: {payload['request_id']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
