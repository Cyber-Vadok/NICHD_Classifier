#!/usr/bin/env python3
# predictor_cli.py
# Interactive terminal classifier for RoBERTa sequence classification models.

import argparse
import sys
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path

CKPT_EXTS = {".bin", ".pt", ".pth", ".safetensors"}

def _infer_num_labels_from_state(state_dict):
    # get num_labels from classifier shape
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and 2 <= v.shape[0] <= 64:
            return v.shape[0]
    return None

def load_model(model_path: str, device: torch.device, base_model: str):
    p = Path(model_path).expanduser()

    # 1) HuggingFace directory type (with config.json)
    if p.is_dir() and (p / "config.json").exists():
        model = AutoModelForSequenceClassification.from_pretrained(str(p))
        model.to(device).eval()
        return model

    # 2) file checkpoint as state_dict
    if p.is_file() and p.suffix.lower() in CKPT_EXTS:
        if p.suffix.lower() == ".safetensors":
            from safetensors.torch import load_file as safe_load_file
            state = safe_load_file(str(p), device=str(device))
        else:
            state = torch.load(str(p), map_location=device)
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError(f"Checkpoint '{p}' non è uno state_dict.")

        num_labels = _infer_num_labels_from_state(state) or 2
        model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_labels)
        model.load_state_dict(state, strict=False)
        model.to(device).eval()
        return model

    # 3) fallback: ID on HF Hub or path
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device).eval()
    return model

def get_logits(model, tokenizer, text: str, device: torch.device, max_length: int = 512):
    enc = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    return logits.squeeze(0).detach().cpu()

def predict_label(model, tokenizer, text: str, device: torch.device, max_length: int = 512):
    logits = get_logits(model, tokenizer, text, device, max_length=max_length)
    pred_id = int(torch.argmax(logits).item())
    # Try to map to a human-readable label, if available
    id2label = getattr(getattr(model, "config", None), "id2label", None)
    if isinstance(id2label, dict) and pred_id in id2label:
        label = id2label[pred_id]
    else:
        label = str(pred_id)
    return label, pred_id, logits.tolist()

def main():
    parser = argparse.ArgumentParser(description="Interactive sentence classifier (RoBERTa).")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./roberta_master2_hf",
        help="Path to the trained model (directory for from_pretrained or file for torch.load). Default: ./roberta_master2_hf",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="roberta_master2_hf",
        help="Tokenizer name or path (e.g., roberta-base or a fine-tuned tokenizer dir).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max sequence length for tokenization. Default: 512",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    parser.add_argument(
    "--base-model",
    type=str,
    default="roberta-base",
    help="Base model usato per inizializzare quando si carica un checkpoint come state_dict.",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    print(f"[info] Using device: {device}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    model = load_model(args.model_path, device, base_model=args.base_model)

    # Print label space if available
    id2label = getattr(getattr(model, "config", None), "id2label", None)
    if isinstance(id2label, dict) and len(id2label) > 0:
        # Ensure numeric sort by key
        try:
            ordered = [id2label[i] for i in sorted(id2label.keys())]
        except Exception:
            ordered = list(id2label.values())
        print(f"[info] Detected labels: {ordered}")
    else:
        num_labels = getattr(getattr(model, "config", None), "num_labels", None)
        if isinstance(num_labels, int):
            print(f"[info] Detected number of labels: {num_labels}")
        else:
            print("[warn] Could not determine label names or count from model config.")

    print("\nType a sentence to classify. Press Enter on an empty line or type 'quit'/'exit' to stop.\n")

    try:
        while True:
            try:
                text = input(">>> ").strip()
            except EOFError:
                # end of file (e.g., piped input finished)
                break
            if text == "" or text.lower() in {"quit", "exit"}:
                break

            label, pred_id, logits = predict_label(model, tokenizer, text, device, max_length=args.max_length)
            print(f"[pred] label: {label} (id={pred_id})")
            print(f"[pred] logits: {logits}\n")

    except KeyboardInterrupt:
        print("\n[info] Interrupted by user. Bye!")
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
