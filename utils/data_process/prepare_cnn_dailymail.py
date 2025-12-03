import argparse
import json
from datasets import load_dataset
from tqdm import tqdm
import os

SYSTEM_PROMPT = "You are a helpful news summarization assistant."

USER_TEMPLATE = (
    "Summarize the following news article in a concise paragraph.\n\n"
    "Article:\n{article}"
)

def example_to_chat(example):
    article = example["article"]
    summary = example["highlights"]

    conv = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(article=article)},
        {"role": "assistant", "content": summary},
    ]

    return {
        "id": example.get("id", ""),
        "conversation": conv,
    }

def main(args):
    # You can also use "ccdv/cnn_dailymail" – they are equivalent in structure.
    dataset = load_dataset(
        "cnn_dailymail",
        "3.0.0",   # good modern version
        split=args.split
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for ex in tqdm(dataset, desc=f"Processing {args.split}"):
            chat_ex = example_to_chat(ex)
            f.write(json.dumps(chat_ex, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"]
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Where to write the JSONL file",
    )
    args = parser.parse_args()
    main(args)
