"""
Example usage of the ImageClassifier plugin with the Maki framework.
"""

import asyncio

from maki import MakiLLama
from maki.plugins.image_classifier import ImageClassifier

# Requires a running Ollama instance with a vision-capable model.
maki = MakiLLama(model="gemma4:26b", base_url="http://localhost:11434")
classifier = ImageClassifier(maki_instance=maki)


def classify_indoor_outdoor(image_path: str) -> str:
    """Return 'indoors', 'outdoors', or 'unknown' for an image file."""
    result = classifier.classify_image(
        image_path=image_path,
        prompt="Is this image taken indoors or outdoors? Reply with one word only: indoors or outdoors.",
        valid_labels=["indoors", "outdoors"],
        fallback_label="unknown",
    )
    if result["success"]:
        return result["label"]
    return f"error: {result['error']}"


async def classify_batch_async(image_paths: list, prompt: str) -> list:
    """Classify multiple images concurrently using async coroutines."""
    tasks = [
        classifier.classify_image_async_coro(image_path=path, prompt=prompt)
        for path in image_paths
    ]
    return await asyncio.gather(*tasks)


if __name__ == "__main__":
    print("ImageClassifier plugin — example usage")
    print("=" * 40)

    # Example 1: Synchronous classification with label allowlist
    print("\nExample 1: Synchronous classification")
    result = classifier.classify_image(
        image_path="sample.jpg",
        prompt="What is the dominant colour in this image? Reply with one word.",
        system="You are a concise image analysis assistant.",
    )
    if result["success"]:
        print(f"  Label: {result['label']}")
    else:
        print(f"  Error: {result['error']}")

    # Example 2: Classification with a strict label allowlist
    print("\nExample 2: Label validation")
    result = classifier.classify_image(
        image_path="sample.jpg",
        prompt="Is this image safe for work? Reply with exactly one word: safe or unsafe.",
        valid_labels=["safe", "unsafe"],
        fallback_label="unknown",
    )
    print(f"  Label: {result['label']}")

    # Example 3: Async wrapper (safe from non-async code)
    print("\nExample 3: Async wrapper from synchronous code")
    result = classifier.classify_image_async(
        image_path="sample.jpg",
        prompt="Does this image contain a person? Reply yes or no.",
        valid_labels=["yes", "no"],
    )
    print(f"  Label: {result['label']}")

    # Example 4: Native async — batch classification
    print("\nExample 4: Concurrent async batch classification")
    images = ["img1.jpg", "img2.jpg", "img3.jpg"]
    results = asyncio.run(
        classify_batch_async(
            image_paths=images,
            prompt="Describe the scene in three words or fewer.",
        )
    )
    for path, res in zip(images, results):
        label = res["label"] if res["success"] else f"error: {res['error']}"
        print(f"  {path}: {label}")
