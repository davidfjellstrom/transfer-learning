"""Train/val/test-split och balanserade delmängder ur data/raw/train."""

import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "train"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

CLASSES = ["0_Recyclable", "1_Electronic", "2_Organic"]

VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42


def class_distribution(raw_dir: Path = RAW_TRAIN_DIR) -> dict[str, int]:
    return {c: len(list((raw_dir / c).glob("*"))) for c in CLASSES}


def create_split(
    raw_dir: Path = RAW_TRAIN_DIR,
    splits_dir: Path = SPLITS_DIR,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = SEED,
) -> None:
    """Skapar data/splits/{train,val,test}/<klass>/ som symlänkar till data/raw/train, stratifierat per klass."""
    rng = random.Random(seed)

    for split in ("train", "val", "test"):
        for c in CLASSES:
            (splits_dir / split / c).mkdir(parents=True, exist_ok=True)

    for c in CLASSES:
        files = sorted((raw_dir / c).glob("*"))
        rng.shuffle(files)
        n_val = int(len(files) * val_frac)
        n_test = int(len(files) * test_frac)

        groups = {
            "val": files[:n_val],
            "test": files[n_val : n_val + n_test],
            "train": files[n_val + n_test :],
        }
        for split, split_files in groups.items():
            for f in split_files:
                link = splits_dir / split / c / f.name
                if not link.exists():
                    link.symlink_to(f.resolve())


def balanced_subset(
    n_per_class: int, split_dir: Path = SPLITS_DIR / "train", seed: int = SEED
) -> dict[str, list[Path]]:
    """Drar n_per_class slumpmässiga (men reproducerbara) filer per klass ur en split-mapp."""
    rng = random.Random(seed)
    subset = {}
    for c in CLASSES:
        files = sorted((split_dir / c).glob("*"))
        if n_per_class > len(files):
            raise ValueError(f"{c} har bara {len(files)} bilder i {split_dir}, kan inte dra {n_per_class}")
        subset[c] = rng.sample(files, n_per_class)
    return subset


def materialize_subset(
    n_per_class: int, out_dir: Path, split_dir: Path = SPLITS_DIR / "train", seed: int = SEED
) -> Path:
    """Skapar en mapp med symlänkar för en balanserad delmängd, redo för flow_from_directory."""
    subset = balanced_subset(n_per_class, split_dir, seed)
    for c, files in subset.items():
        (out_dir / c).mkdir(parents=True, exist_ok=True)
        for f in files:
            link = out_dir / c / f.name
            if not link.exists():
                link.symlink_to(f.resolve())
    return out_dir


def plot_class_distribution(
    dist: dict[str, int], out_path: Path = FIGURES_DIR / "class_distribution.png"
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(dist.keys(), dist.values())
    ax.set_ylabel("Antal bilder")
    ax.set_title("Klassbalans i träningsdata")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_class_examples(
    raw_dir: Path = RAW_TRAIN_DIR,
    n_per_class: int = 4,
    seed: int = SEED,
    out_path: Path = FIGURES_DIR / "class_examples.png",
) -> Path:
    rng = random.Random(seed)
    fig, axes = plt.subplots(len(CLASSES), n_per_class, figsize=(n_per_class * 2, len(CLASSES) * 2))
    for row, c in enumerate(CLASSES):
        files = rng.sample(sorted((raw_dir / c).glob("*")), n_per_class)
        for col, f in enumerate(files):
            ax = axes[row][col]
            ax.imshow(Image.open(f))
            ax.axis("off")
            if col == 0:
                ax.set_title(c, fontsize=10, loc="left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    dist = class_distribution()
    print("Klassbalans i data/raw/train:")
    for c, n in dist.items():
        print(f"  {c}: {n}")

    create_split()
    print(f"Split skapad i {SPLITS_DIR}")

    dist_path = plot_class_distribution(dist)
    examples_path = plot_class_examples()
    print(f"Sparade {dist_path} och {examples_path}")
