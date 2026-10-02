# openai-image

A standalone, dependency-free Python CLI for generating and editing images with the OpenAI image API.

- Defaults to `gpt-image-2.5-flare`
- Supports `gpt-image-2.5-sunburst` for maximum quality and editing precision
- Accepts local paths, URLs, data URLs, and OpenAI file IDs as references
- Supports masks, 1–16 references, up to 10 outputs, custom dimensions, transparent backgrounds, and quality through `max`
- Saves image files plus a metadata sidecar; never prints raw image base64
- Offers human-readable and JSON output

## Install

From a checkout:

```bash
uv tool install .
```

Directly from GitHub:

```bash
uv tool install git+https://github.com/almostmachines/openai-image.git
```

Once published to PyPI:

```bash
uv tool install openai-image
```

To run without installing:

```bash
uvx --from . openai-image --help
```

Set your API key:

```bash
export OPENAI_API_KEY="sk-..."
```

## Examples

```bash
# Validate and inspect a request without making an API call
openai-image --dry-run --prompt "A tiny owl reading a book"

# Generate an image
openai-image --prompt "A tiny owl reading a book" --quality low

# Generate from a reference
openai-image \
  --prompt "Create a polished product hero shot using the reference" \
  --reference ./product.png \
  --size 1536x1024

# Use multiple references
openai-image -p "Combine these into a cohesive editorial illustration" \
  -r ./subject.png -r https://example.com/style.png

# Precision edit with a mask
openai-image \
  --prompt "Keep the complete scene unchanged except replace the masked vase with blue glass" \
  --reference ./room.png \
  --mask ./mask.png \
  --model gpt-image-2.5-sunburst \
  --quality high

# Transparent WebP
openai-image -p "A clean isolated product icon" \
  --background transparent --format webp

# Machine-readable request and response
openai-image --input request.json --json
cat request.json | openai-image --input - --json
```

Images are written to `./openai-images` by default. Change this with `--output-dir` or `OPENAI_IMAGE_OUTPUT_DIR`.

## Options

Run `openai-image --help` for the complete option list. Key options include:

```text
-p, --prompt TEXT
    --prompt-file PATH
-r, --reference PATH_OR_URL   (repeat up to 16 times)
    --mask PATH_OR_URL
-n COUNT                      (1-10)
    --size WIDTHxHEIGHT       (or auto)
    --quality VALUE           (low, medium, high, xhigh, max, auto)
    --format VALUE            (png, jpeg, webp)
    --compression N           (0-100; JPEG/WebP only)
    --background VALUE        (auto, opaque, transparent)
    --moderation VALUE        (auto, low)
    --model MODEL
-o, --output-dir PATH
    --output-prefix NAME
    --transport VALUE         (auto, json, multipart)
    --dry-run
    --no-metadata
    --json
```

For GPT Image 2/2.5 custom dimensions, both edges must be divisible by 16, the aspect ratio must be between 1:3 and 3:1, and the image must contain 655,360–8,294,400 pixels. The maximum edge is 3840 px.

## JSON input

CLI options override corresponding values from the input object. CLI references are appended to references from the object.

```json
{
  "prompt": "A detailed final-image prompt",
  "n": 1,
  "size": "1024x1024",
  "model": "gpt-image-2.5-flare",
  "quality": "medium",
  "output_format": "png",
  "reference_images": ["./reference.png"],
  "mask": "./mask.png",
  "output_dir": "./openai-images",
  "output_prefix": "my-image"
}
```

## References and transports

The CLI automatically chooses:

- JSON for generations
- Multipart form data for edits using only local files
- JSON for edits containing URLs, data URLs, or OpenAI file IDs

Use `--transport json` or `--transport multipart` to override this. Local images are embedded as data URLs when JSON transport is used. Multipart transport requires all references and the mask to be local files.

A leading `@` on a local path is accepted and removed. This makes both `image.png` and `@image.png` valid.

## Output

Each request receives a timestamped, prompt-derived filename unless `--output-prefix` is supplied. Existing files are never overwritten. A JSON metadata sidecar records the sanitized API response, request ID, usage, parameters, and output paths without retaining returned base64 data or image URLs.

Use `--no-metadata` to disable the sidecar.

## Environment

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | API key; required except for dry runs |
| `OPENAI_BASE_URL` | API base URL; defaults to `https://api.openai.com/v1` |
| `OPENAI_ORGANIZATION` | Optional organization header |
| `OPENAI_PROJECT` | Optional project header |
| `OPENAI_IMAGE_OUTPUT_DIR` | Default output directory |
| `OPENAI_IMAGE_TIMEOUT` | HTTP timeout in seconds; defaults to 600 |

`FAITH_IMAGE_OUTPUT_DIR` is also recognized as a compatibility fallback.

## Python API

```python
from openai_image import create_image

result = create_image(
    {
        "prompt": "A tiny owl reading a book",
        "quality": "medium",
        "output_dir": "./openai-images",
    }
)

print(result["files"][0]["path"])
```

## Development

```bash
uv sync
uv run pytest
uv build
```

## License

MIT
