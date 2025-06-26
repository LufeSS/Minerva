import json
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import typer
from rich.console import Console
from tqdm import tqdm

from minerva.data.wikitext import build_dataloader, load_wikitext
from minerva.model import Decoder

console = Console()


def shift_labels(inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return *inputs[:-1]* as model input and *inputs[1:]* as labels."""

    return inputs[:, :-1].contiguous(), inputs[:, 1:].contiguous()


app = typer.Typer(add_completion=False, help="Evaluate a trained Minerva checkpoint on WikiText.")


@app.command()
def main(
    checkpoint: Path = typer.Option(..., "--checkpoint", "-ckpt", help="Path to .pt checkpoint file."),
    seq_len: int = typer.Option(128, help="Sequence length used during training."),
    batch_size: int = typer.Option(8, help="Batch size for evaluation."),
    dataset: str = typer.Option("wikitext-2-raw-v1", help="Dataset variant."),
    split: str = typer.Option("test", help="Dataset split to evaluate (validation | test)."),
    # Model hyper-params (should match those used for training)
    num_layers: int = typer.Option(4, help="Number of decoder layers."),
    hidden_dim: int = typer.Option(512, help="Model hidden dimension."),
    num_heads: int = typer.Option(8, help="Number of attention heads."),
    dropout: float = typer.Option(0.1, help="Dropout rate (kept for completeness)."),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Computation device."),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to trainer_config.json (overrides CLI hyper-parameters).",
    ),
):
    """Compute token-level negative log-likelihood and perplexity."""

    # ------------------------------------------------------------------ #
    # Resolve device
    device = torch.device(device)
    console.print(f"Evaluating on [bold]{device}[/bold]")

    # ------------------------------------------------------------------ #
    # Load training config (if provided) to guarantee architecture match
    if config_file and config_file.exists():
        console.print(f"Loading model hyper-parameters from {config_file}")
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        seq_len = cfg.get("seq_len", seq_len)
        batch_size = cfg.get("batch_size", batch_size)
        dataset = cfg.get("dataset", dataset)
        num_layers = cfg.get("num_layers", num_layers)
        hidden_dim = cfg.get("hidden_dim", hidden_dim)
        num_heads = cfg.get("num_heads", num_heads)
        dropout = cfg.get("dropout", dropout)

    # ------------------------------------------------------------------ #
    # Dataset
    console.print(
        f"Loading WikiText (split={split}, seq_len={seq_len}, variant={dataset})…"
    )
    data_ds, vocab_size, _ = load_wikitext(
        split, block_size=seq_len, dataset_variant=dataset
    )
    data_loader = build_dataloader(data_ds, batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------ #
    # Model
    console.print("Building model…")
    model = Decoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
    )

    # Use DataParallel if multiple GPUs
    if torch.cuda.device_count() > 1 and device.type == "cuda":
        console.print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    # Load checkpoint
    console.print(f"Loading checkpoint from {checkpoint}")
    state_dict = torch.load(checkpoint, map_location="cpu")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Could be a DataParallel checkpoint – strip 'module.' prefix
        cleaned = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned)

    model.to(device)
    model.eval()

    # ------------------------------------------------------------------ #
    # Evaluation loop
    console.print("Starting evaluation…")
    nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            inp, tgt = shift_labels(input_ids)
            logits = model(inp)
            loss_sum = F.cross_entropy(
                logits.view(-1, vocab_size), tgt.view(-1), reduction="sum"
            )
            nll += loss_sum.item()
            total_tokens += tgt.numel()

    avg_nll = nll / total_tokens
    ppl = math.exp(avg_nll)

    console.rule("[bold green]Results")
    console.print(f"Negative log-likelihood: {avg_nll:.4f}")
    console.print(f"Perplexity: {ppl:.2f}")


if __name__ == "__main__":
    app() 