import math
import re
import sys
from pathlib import Path
from typing import Optional

import torch
import typer
from rich.console import Console
from transformers import AutoTokenizer

from minerva.model import Decoder

console = Console()

def load_checkpoint(
    model: torch.nn.Module, checkpoint: Path, map_location: str = "cpu"
) -> None:
    state = torch.load(checkpoint, map_location=map_location)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        # handle DataParallel checkpoints
        cleaned = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(cleaned)


app = typer.Typer(add_completion=False, help="Generate text with a trained Minerva checkpoint.")


@app.command()
def main(
    prompt: str = typer.Argument("Once upon a time", help="Prompt text to start generation."),
    max_new_tokens: int = typer.Option(50, help="Number of tokens to generate."),
    checkpoint: Optional[Path] = typer.Option(
        None,
        "--checkpoint",
        "-ckpt",
        help="Path to a .pt checkpoint. If omitted, the latest decoder_epoch*.pt in --checkpoint-dir is used.",
    ),
    checkpoint_dir: Path = typer.Option("checkpoints", help="Directory containing checkpoints."),
    # Model arch params (should match training)
    num_layers: int = typer.Option(4, help="Number of decoder layers."),
    hidden_dim: int = typer.Option(512, help="Hidden size."),
    num_heads: int = typer.Option(8, help="Attention heads."),
    dropout: float = typer.Option(0.1, help="Dropout."),
    seq_len: int = typer.Option(128, help="Max sequence length during training."),
    tokenizer_name: str = typer.Option("gpt2", help="HF tokenizer name or path."),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu", help="Device to run on."),
    temperature: float = typer.Option(1.0, help="Softmax temperature."),
    top_p: float = typer.Option(0.9, help="Nucleus sampling p value."),
):
    """Generate *max_new_tokens* continuations from *prompt*."""

    device = torch.device(device)
    console.print(f"Using device [bold]{device}[/bold]")

    # ---------------- tokenizer ---------------- #
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    # ---------------- model -------------------- #
    model = Decoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)

    if checkpoint is None:
        pattern = re.compile(r"decoder_epoch(\d+)\.pt")
        candidates = [p for p in checkpoint_dir.glob("decoder_epoch*.pt") if pattern.search(p.name)]
        if not candidates:
            console.print(f"[red]No checkpoint found in {checkpoint_dir}[/red]")
            sys.exit(1)
        candidates.sort(key=lambda p: int(pattern.search(p.name).group(1)))
        checkpoint = candidates[-1]
        console.print(f"[yellow]Auto-selected checkpoint {checkpoint.name}[/yellow]")

    console.print(f"Loading weights from {checkpoint}")
    load_checkpoint(model, checkpoint, map_location="cpu")
    model.eval()

    # ---------------- generation loop ---------------- #
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(generated[:, -seq_len:])  # ensure within trained context
        next_logits = logits[:, -1, :] / temperature  # (1, V)

        # ------------- nucleus (top-p) filtering ------------- #
        if top_p < 1.0:
            probs = torch.softmax(next_logits, dim=-1)  # (1, V)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Mask tokens with cumulative prob above top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            # Ensure at least one token is kept
            sorted_indices_to_remove[..., 0] = False

            # Create a copy so we don't modify probs in-place before sampling
            filtered_probs = probs.clone()
            filtered_probs.scatter_(1, sorted_indices[sorted_indices_to_remove], 0.0)
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
            next_token = torch.multinomial(filtered_probs, num_samples=1)
        else:
            # Pure temperature sampling
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

    console.rule("[bold green]Generated text")
    console.print(tokenizer.decode(generated[0]))


if __name__ == "__main__":
    app() 