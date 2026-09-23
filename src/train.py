"""LoRA/SFT fine-tuning of the base model on the listing-extraction task. Run on a GPU (Colab/Kaggle)."""

import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Mirrors CONFIG in notebooks/main.ipynb (section 2) — keep these two in sync.
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CATEGORY = "Computers And Accessories"
MAX_SEQ_LENGTH = 512
SEED = 42
ADAPTER_OUT = MODELS_DIR / "qwen2.5-0.5b-listing-extract-lora"


def load_split(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_attributes(rows: list[dict]) -> list[str]:
    attrs: set[str] = set()
    for r in rows:
        attrs.update(r["gold_json"].keys())
    return sorted(attrs)


def build_messages(title: str, description: str, attributes: list[str], gold_json: dict | None = None) -> list[dict]:
    system_prompt = (
        "You are a product-listing extraction assistant. Given a listing title and description, "
        "extract values for the following attributes: " + ", ".join(attributes) + ". "
        "Respond with a single JSON object using exactly these keys. "
        "Use null for any attribute that isn't mentioned in the text."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Title: {title}\nDescription: {description}"},
    ]
    if gold_json is not None:
        messages.append({"role": "assistant", "content": json.dumps(gold_json)})
    return messages


def build_dataset(rows: list[dict], attributes: list[str]) -> "Dataset":
    examples = [
        {"messages": build_messages(r["input_title"], r["input_description"], attributes, r["gold_json"])}
        for r in rows
    ]
    return Dataset.from_list(examples)


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto", device_map="auto")
    return model, tokenizer


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )


def build_training_config() -> SFTConfig:
    has_cuda = torch.cuda.is_available()
    return SFTConfig(
        output_dir=str(MODELS_DIR / "checkpoints"),
        max_length=MAX_SEQ_LENGTH,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        seed=SEED,
        report_to="none",
        use_cpu=not has_cuda,
        bf16=has_cuda,
    )


def main() -> None:
    train_rows = [r for r in load_split("train") if r["category"] == CATEGORY]
    val_rows = [r for r in load_split("val") if r["category"] == CATEGORY]
    attributes = get_attributes(train_rows)
    print(f"{len(train_rows)} train / {len(val_rows)} val rows, {len(attributes)} attributes")

    train_dataset = build_dataset(train_rows, attributes)
    val_dataset = build_dataset(val_rows, attributes)

    model, tokenizer = load_model_and_tokenizer()

    trainer = SFTTrainer(
        model=model,
        args=build_training_config(),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=build_lora_config(),
        processing_class=tokenizer,
    )

    trainer.train()

    ADAPTER_OUT.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ADAPTER_OUT))
    print(f"Adapter saved to {ADAPTER_OUT}")


if __name__ == "__main__":
    main()
