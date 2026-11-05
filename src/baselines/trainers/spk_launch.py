from __future__ import annotations

def ensure_schnetpack_importable() -> None:
    import torch.utils.data.dataloader as dl

    if not hasattr(dl, "T_co") and hasattr(dl, "_T_co"):
        dl.T_co = dl._T_co


def main() -> None:
    ensure_schnetpack_importable()
    from schnetpack.cli import train

    train()


if __name__ == "__main__":
    main()
