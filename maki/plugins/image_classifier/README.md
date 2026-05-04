# Image Classifier Plugin for Maki Framework

Classifies images using a vision-capable Ollama model (e.g. `gemma4:26b`).
The prompt and optional system prompt are provided by the caller, making the
plugin domain-agnostic and reusable for any classification task.

## Features

- Single-image classification (synchronous and asynchronous)
- Optional label allowlist — responses outside the list fall back to a default
- Base64 image encoding handled internally
- Works with any vision-capable Ollama model

## Requirements

A running Ollama instance with a vision-capable model loaded:

```bash
ollama pull gemma4:26b
```

## Usage

```python
from maki.maki import Maki
from maki.plugins.image_classifier import ImageClassifier

maki = Maki("http://localhost", 11434, "gemma4:26b")
classifier = ImageClassifier(maki_instance=maki)
```

### `classify_image(image_path, prompt, system=None, valid_labels=None, fallback_label="unknown")`

Classifies a single image synchronously.

```python
result = classifier.classify_image(
    image_path="photo.jpg",
    prompt="Is this image indoors or outdoors? Reply with one word only.",
    valid_labels=["indoors", "outdoors"],
    fallback_label="unknown",
)

if result["success"]:
    print(result["label"])   # "indoors" or "outdoors"
else:
    print(result["error"])
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `image_path` | `str` | Path to the image file |
| `prompt` | `str` | Instruction sent to the model |
| `system` | `str \| None` | Optional system prompt |
| `valid_labels` | `list[str] \| None` | Allowed labels; `None` = accept any response |
| `fallback_label` | `str` | Used when the model returns an unrecognised label |

**Returns:** `dict` with keys `success` (bool), `label` (str), `image_path` (str), `error` (str or None).

---

### `classify_image_async(image_path, prompt, ...)`

Synchronous wrapper around the async implementation — safe to call from
non-async code without managing an event loop manually.

```python
result = classifier.classify_image_async(
    image_path="photo.jpg",
    prompt="Describe the dominant colour in one word.",
)
```

---

### `classify_image_async_coro(image_path, prompt, ...)`

Native async coroutine — `await` it directly inside an `asyncio` event loop.

```python
import asyncio

async def run():
    result = await classifier.classify_image_async_coro(
        image_path="photo.jpg",
        prompt="What object is most prominent? Reply with one word.",
        valid_labels=["car", "person", "building", "tree"],
    )
    print(result["label"])

asyncio.run(run())
```

---

## Error Handling

All methods return a dict with `success: False` and an `error` field when
classification fails. Common failure cases:

- `image_path` does not exist — `FileNotFoundError`
- No Maki instance configured — returns error immediately without a network call
- Model returns a response not in `valid_labels` — label is replaced with `fallback_label`

## Integration with Maki Agents

```python
from maki.maki import Maki
from maki.plugins.image_classifier import ImageClassifier

maki = Maki("http://localhost", 11434, "gemma4:26b")
classifier = ImageClassifier(maki_instance=maki)

image_files = ["img1.jpg", "img2.png", "img3.webp"]
labels = ["safe", "unsafe", "unknown"]

for path in image_files:
    result = classifier.classify_image(
        image_path=path,
        prompt="Is this image safe for work? Reply: safe or unsafe.",
        valid_labels=labels,
    )
    print(f"{path}: {result['label']}")
```
