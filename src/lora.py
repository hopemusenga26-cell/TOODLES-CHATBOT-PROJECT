"""Optional LoRA fine-tuning, adapter compatibility checks and adapter merging (task 7).

Run only after the retrieval chatbot works. The base model, its revision and the LoRA
settings are in the `lora` section of config/config.yaml; the training split is built by
src.lora_data.

Usage:
    python -m src.lora train --group A --name lora_A        # train on group A's examples
    python -m src.lora check artifacts/lora/lora_A artifacts/lora/lora_B
    python -m src.lora merge artifacts/lora/lora_A artifacts/lora/lora_B --name merged_AB
Then evaluate an adapter like any other answer model:
    python -m src.evaluate --split validation --backend hf --lora-base --adapter artifacts/lora/lora_A

Adapters can only be combined when they were trained on the same base model, revision,
architecture and tokenizer, with the same kind of adapter on the same layers; `check` compares
all of these before `merge` will run. A merged adapter is evaluated like any other: it is only
worth adopting if it does not lose accuracy compared with the separate adapters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import load_config, read_json, read_jsonl, resolve, write_json

RECORD = "training_record.json"
MERGE_RECORD = "merge_record.json"
PROMPT_VERSION = "v2"  # bump when SYSTEM_PROMPT, INSTRUCTIONS or build_prompt change


# ---------------------------------------------------------------- compatibility checks

def adapter_card(adapter_dir: str | Path) -> dict:
    """Everything that decides whether an adapter can be combined with another one."""
    adapter_dir = Path(adapter_dir)
    config = read_json(adapter_dir / "adapter_config.json")
    if config is None:
        raise FileNotFoundError(f"No adapter_config.json in {adapter_dir}")
    record = read_json(adapter_dir / RECORD, default={}) or read_json(adapter_dir / MERGE_RECORD, default={})
    base = record.get("base_model", {})
    return {
        "path": str(adapter_dir),
        "base_model": config.get("base_model_name_or_path"),
        "base_revision": config.get("revision") or base.get("revision"),
        "architecture": base.get("architecture"),
        "tokenizer_sha256": base.get("tokenizer_sha256"),
        "peft_type": config.get("peft_type"),
        "task_type": config.get("task_type"),
        "target_modules": sorted(config.get("target_modules") or []),
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "use_rslora": bool(config.get("use_rslora")),
        "use_dora": bool(config.get("use_dora")),
        "prompt_version": record.get("prompt_version"),
    }


def compatibility(a: dict, b: dict) -> tuple[list[str], list[str]]:
    """Return (blocking problems, warnings) for combining adapter cards `a` and `b`."""
    problems, warnings = [], []
    must_match = {
        "base_model": "base model",
        "base_revision": "base model revision",
        "architecture": "model architecture",
        "tokenizer_sha256": "tokenizer",
        "peft_type": "adapter type",
        "task_type": "task type",
        "target_modules": "adapted layers (target modules)",
        "use_dora": "DoRA setting",
    }
    for key, label in must_match.items():
        if a.get(key) is None or b.get(key) is None:
            problems.append(f"{label} is not recorded for both adapters, so compatibility cannot be confirmed.")
        elif a[key] != b[key]:
            problems.append(f"Different {label}: {a[key]} vs {b[key]}.")
    if a.get("r") != b.get("r"):
        warnings.append(f"Different ranks ({a.get('r')} vs {b.get('r')}): only the 'cat' or 'svd' merge methods apply.")
    if a.get("lora_alpha") != b.get("lora_alpha") or a.get("use_rslora") != b.get("use_rslora"):
        warnings.append("Different scaling (lora_alpha or rsLoRA): each update keeps its own scaling when merged with 'cat'.")
    if a.get("prompt_version") != b.get("prompt_version"):
        warnings.append(f"Trained with different prompt versions ({a.get('prompt_version')} vs {b.get('prompt_version')}).")
    return problems, warnings


def check_adapter_base(adapter_dir: str | Path, model_name: str, revision: str | None) -> None:
    """Refuse to load an adapter on a different base model or revision than it was trained on."""
    card = adapter_card(adapter_dir)
    if card["base_model"] != model_name or (revision and card["base_revision"] not in (None, revision)):
        raise ValueError(f"Adapter {adapter_dir} was trained on {card['base_model']}@{card['base_revision']}, "
                         f"not {model_name}@{revision}.")


# ---------------------------------------------------------------- training

def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def base_model_info(model, tokenizer, name: str, revision: str) -> dict:
    from huggingface_hub import hf_hub_download

    tok_file = hf_hub_download(name, "tokenizer.json", revision=revision)
    c = model.config
    return {
        "name": name,
        "revision": revision,
        "architecture": (c.architectures or [type(model).__name__])[0],
        "layers": c.num_hidden_layers,
        "hidden_size": c.hidden_size,
        "attention_heads": c.num_attention_heads,
        "key_value_heads": getattr(c, "num_key_value_heads", None),
        "vocab_size": c.vocab_size,
        "parameters": sum(p.numel() for p in model.parameters()),
        "tokenizer_sha256": sha256_file(tok_file),
        "licence": "Apache-2.0",
    }


def encode_example(tokenizer, example: dict, passages: dict, max_length: int):
    """Token ids for prompt + answer, and where the answer starts; None if too long."""
    from src.generate import SYSTEM_PROMPT, build_prompt

    evidence = [passages[c] for c in example["evidence"]]
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(example["question"], evidence)}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": example["target"]}],
                                         tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("The chat template does not extend the prompt; cannot locate the answer tokens.")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(full[len(prompt):].rstrip("\n"), add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(answer_ids) > max_length:
        return None
    return prompt_ids + answer_ids, len(prompt_ids)


def answer_loss(model, ids: list[int], start: int):
    """Cross-entropy on the answer tokens only; logits are computed only where they are needed."""
    import torch

    input_ids = torch.tensor([ids])
    n = len(ids) - start
    logits = model(input_ids=input_ids, logits_to_keep=n + 1).logits[0, :-1]
    return torch.nn.functional.cross_entropy(logits.float(), input_ids[0, start:])


def train(group: str, name: str, cfg: dict, limit: int | None = None) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.evaluate import peak_ram_mb
    from src.lora_data import OUTPUT as DATA_FILE

    lc = cfg["lora"]
    random.seed(lc["seed"])
    torch.manual_seed(lc["seed"])

    data_path = resolve(DATA_FILE)
    examples = [e for e in read_jsonl(data_path) if group == "all" or e["group"] == group]
    if limit:
        examples = examples[:limit]
    if not examples:
        raise SystemExit(f"No training examples for group {group}; run python -m src.lora_data first.")
    passages = {p["chunk_id"]: p for p in read_jsonl(resolve(cfg["paths"]["index_dir"]) / "passages.jsonl")}

    tokenizer = AutoTokenizer.from_pretrained(lc["base_model"], revision=lc["base_revision"])
    model = AutoModelForCausalLM.from_pretrained(lc["base_model"], revision=lc["base_revision"], dtype=torch.float32)
    base = base_model_info(model, tokenizer, lc["base_model"], lc["base_revision"])
    model.config.use_cache = False
    lora_config = LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                             target_modules=list(lc["target_modules"]), bias="none", task_type="CAUSAL_LM",
                             revision=lc["base_revision"])
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    encoded, too_long = [], 0
    for e in examples:
        item = encode_example(tokenizer, e, passages, lc["max_length"])
        if item is None:
            too_long += 1
        else:
            encoded.append((e["id"], e["kind"], *item))
    # A small held-out slice of the training data tracks overfitting (the evaluation splits are not used).
    random.shuffle(encoded)
    n_dev = max(4, round(lc["dev_share"] * len(encoded)))
    dev, train_set = encoded[:n_dev], encoded[n_dev:]

    def dev_loss() -> float:
        model.eval()
        with torch.no_grad():
            losses = [answer_loss(model, ids, start).item() for _, _, ids, start in dev]
        model.train()
        return round(sum(losses) / len(losses), 4)

    accum, epochs = lc["grad_accum"], lc["epochs"]
    total_updates = math.ceil(len(train_set) / accum) * epochs
    warmup = max(1, round(0.06 * total_updates))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=lc["learning_rate"], weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda u: min(1.0, (u + 1) / warmup) * max(
        0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, u / total_updates)))))

    history = {"dev_loss": [{"epoch": 0, "loss": dev_loss()}], "train_loss": []}
    print(f"Group {group}: {len(train_set)} training and {len(dev)} held-out examples "
          f"({too_long} too long), {trainable:,} trainable parameters, {total_updates} updates. "
          f"Held-out loss before training: {history['dev_loss'][0]['loss']}", flush=True)
    start_time, update, running = time.time(), 0, []
    model.train()
    for epoch in range(1, epochs + 1):
        random.shuffle(train_set)
        for i, (_, _, ids, start) in enumerate(train_set, 1):
            loss = answer_loss(model, ids, start)
            (loss / accum).backward()
            running.append(loss.item())
            if i % accum == 0 or i == len(train_set):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                schedule.step()
                optimizer.zero_grad()
                update += 1
                history["train_loss"].append({"update": update, "loss": round(sum(running) / len(running), 4)})
                running = []
                if update % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"  update {update}/{total_updates}, loss {history['train_loss'][-1]['loss']}, "
                          f"{elapsed / 60:.0f} min, about {elapsed / update * (total_updates - update) / 60:.0f} min left",
                          flush=True)
        history["dev_loss"].append({"epoch": epoch, "loss": dev_loss()})
        print(f"  epoch {epoch}: held-out loss {history['dev_loss'][-1]['loss']}", flush=True)

    out = resolve(lc["output_dir"]) / name
    model.save_pretrained(out, save_embedding_layers=False)
    (out / "README.md").unlink(missing_ok=True)  # PEFT's empty model-card template; training_record.json replaces it
    import peft
    import transformers

    write_json({
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Teach the small base model the chatbot's answer format: short answers from the numbered "
                   "passages with [n] citations, or INSUFFICIENT_EVIDENCE when the passages do not cover the question.",
        "base_model": base,
        "prompt_version": PROMPT_VERSION,
        "lora": {k: lc[k] for k in ("r", "alpha", "dropout", "target_modules")},
        "trainable_parameters": trainable,
        "training": {
            "group": group,
            "data_file": DATA_FILE,
            "data_sha256": sha256_file(data_path),
            "examples_used": len(train_set),
            "held_out_examples": len(dev),
            "skipped_too_long": too_long,
            "by_kind": {k: sum(e[1] == k for e in train_set) for k in ("answer", "insufficient")},
            "epochs": epochs,
            "gradient_accumulation": accum,
            "optimizer_updates": update,
            "learning_rate": lc["learning_rate"],
            "schedule": f"linear warm-up over {warmup} updates, then cosine decay",
            "max_length": lc["max_length"],
            "loss": "cross-entropy on answer tokens only",
            "precision": "float32",
            "seed": lc["seed"],
            "device": "cpu",
            "minutes": round((time.time() - start_time) / 60, 1),
            "peak_ram_mb": peak_ram_mb(),
        },
        "history": history,
        "software": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "peft": peft.__version__},
        "licence_and_sharing": "Base model Apache-2.0. The adapter was trained on passages from the group's books; "
                               "share it only through the project's private storage, like the index.",
    }, out / RECORD)
    print(f"Saved adapter to {out}")
    return out


# ---------------------------------------------------------------- merging

def merge(adapter_dirs: list[str], weights: list[float], name: str, method: str, cfg: dict) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    cards = [adapter_card(d) for d in adapter_dirs]
    for other in cards[1:]:
        problems, warnings = compatibility(cards[0], other)
        for w in warnings:
            print(f"warning: {w}")
        if problems:
            raise SystemExit("Not compatible, not merging:\n  " + "\n  ".join(problems))
    lc = cfg["lora"]
    model = AutoModelForCausalLM.from_pretrained(cards[0]["base_model"], revision=cards[0]["base_revision"],
                                                 dtype=torch.float32)
    names = [f"a{i}" for i in range(len(adapter_dirs))]
    model = PeftModel.from_pretrained(model, adapter_dirs[0], adapter_name=names[0])
    for n, d in zip(names[1:], adapter_dirs[1:]):
        model.load_adapter(d, adapter_name=n)
    # 'cat' stacks the low-rank factors, so the merged update is exactly sum(weight_i * delta_W_i).
    model.add_weighted_adapter(names, weights, adapter_name=name, combination_type=method)
    out_root = resolve(lc["output_dir"])
    model.save_pretrained(out_root, selected_adapters=[name], save_embedding_layers=False)
    out = out_root / name
    for card in (out_root / "README.md", out / "README.md"):  # PEFT's empty model-card template
        card.unlink(missing_ok=True)
    first = read_json(Path(adapter_dirs[0]) / RECORD, default={})
    write_json({
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": [{"path": str(Path(d).as_posix()), "weight": w} for d, w in zip(adapter_dirs, weights)],
        "method": method,
        "formula": "W = W0 + sum_i(weight_i * delta_W_i)" if method == "cat" else f"PEFT '{method}' combination",
        "base_model": first.get("base_model", {}),
        "prompt_version": first.get("prompt_version"),
        "compatibility_checked": [Path(card["path"]).as_posix() for card in cards],
    }, out / MERGE_RECORD)
    print(f"Saved merged adapter to {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="train a LoRA adapter")
    t.add_argument("--group", choices=["A", "B", "all"], default="all")
    t.add_argument("--name", required=True)
    t.add_argument("--limit", type=int, help="use only the first N examples (for a quick trial)")
    c = sub.add_parser("check", help="check whether adapters can be combined")
    c.add_argument("adapters", nargs="+")
    m = sub.add_parser("merge", help="combine compatible adapters")
    m.add_argument("adapters", nargs="+")
    m.add_argument("--weights", type=float, nargs="+", help="one per adapter (default: equal, summing to 1)")
    m.add_argument("--method", default="cat", choices=["cat", "linear", "svd", "ties"])
    m.add_argument("--name", required=True)
    args = parser.parse_args(argv)
    cfg = load_config()

    if args.command == "train":
        train(args.group, args.name, cfg, args.limit)
    elif args.command == "check":
        cards = [adapter_card(d) for d in args.adapters]
        print(json.dumps(cards, indent=2))
        ok = True
        for other in cards[1:]:
            problems, warnings = compatibility(cards[0], other)
            print(f"\n{cards[0]['path']} + {other['path']}:")
            for p in problems:
                print(f"  NOT COMPATIBLE: {p}")
            for w in warnings:
                print(f"  warning: {w}")
            if not problems:
                print("  compatible")
            ok = ok and not problems
        return 0 if ok else 1
    else:
        weights = args.weights or [1.0 / len(args.adapters)] * len(args.adapters)
        if len(weights) != len(args.adapters):
            parser.error("give one weight per adapter")
        merge(args.adapters, weights, args.name, args.method, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
