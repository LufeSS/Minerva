import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from rich.console import Console

from minerva.model import Decoder
from minerva.data.wikitext import load_wikitext, build_dataloader

console = Console()

def parse_args():
    p = argparse.ArgumentParser(description="Train Minerva TR1 on WikiText-2")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--expansion", type=int, default=4)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    return p.parse_args()

def shift_labels(inputs: torch.Tensor):
    # Returns (model_inputs, targets)
    return inputs[:, :-1].contiguous(), inputs[:, 1:].contiguous()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    console.print(f"[bold]Loading dataset[/bold] (seq_len={args.seq_len})…")
    train_ds, vocab_size, tokenizer = load_wikitext("train", block_size=args.seq_len)
    val_ds, _, _ = load_wikitext("validation", block_size=args.seq_len)

    train_loader = build_dataloader(train_ds, args.batch_size, shuffle=True)
    val_loader = build_dataloader(val_ds, args.batch_size, shuffle=False)

    console.print("Building model…")
    model = Decoder(
        vocab_size=vocab_size,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ---------- train ----------
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            input_ids = batch["input_ids"].to(device)
            targets = input_ids.clone()
            inp, tgt = shift_labels(targets)
            logits = model(inp)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size), tgt.view(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_ppl = math.exp(running_loss / len(train_loader))

        # ---------- validation ----------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                inp, tgt = shift_labels(input_ids)
                logits = model(inp)
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size), tgt.view(-1)
                )
                val_loss += loss.item()
        val_ppl = math.exp(val_loss / len(val_loader))
        console.print(
            f"[green]Epoch {epoch}: train ppl {train_ppl:.2f}, val ppl {val_ppl:.2f}[/green]"
        )

        # save ckpt
        torch.save(model.state_dict(), ckpt_dir / f"decoder_epoch{epoch}.pt")

if __name__ == "__main__":
    main() 