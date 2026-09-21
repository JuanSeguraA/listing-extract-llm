# listing-extract-llm

Fine-tuning a small open-source LLM to extract structured attribute-value pairs (JSON) from raw marketplace product listings — titles and descriptions — the kind of task that powers faceted search, auto-categorization, and listing auto-fill on a classifieds/marketplace platform.

## Dataset

[WDC PAVE](https://webdatacommons.org/structureddata/wdc-pave/) — 1,420 human-annotated product offers across 5 categories (Computers & Accessories, Home & Garden, Office Products, Grocery & Gourmet Food, Jewelry), with ~24.5k annotated attribute-value pairs. Ready-made train/test JSONL splits, no scraping required.

The accompanying paper reports GPT-3.5/GPT-4 F1 scores (~79-91%) on extraction — a baseline to compare against.

