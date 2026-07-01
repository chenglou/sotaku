import os
from pathlib import Path


TOKEN = "SOTAKU_SHARED_CACHE_LINE_"


def read_dataset_path(env_name: str, fallback: Path) -> list[str]:
    path = Path(os.environ.get(env_name, fallback))
    if not path.exists():
        return []
    return path.read_text().splitlines()


def main() -> None:
    train_lines = read_dataset_path("VD_DATA_TRAIN", Path("data/train.txt"))
    val_lines = read_dataset_path("VD_DATA_VAL", Path("data/val.txt"))

    train_ok = all(line.startswith(TOKEN) for line in train_lines) and bool(train_lines)
    val_ok = all(line.startswith(TOKEN) for line in val_lines) and bool(val_lines)
    no_test_file = not Path("data/test.txt").exists()

    if train_ok and val_ok and no_test_file:
        score = len(train_lines) * 1000 + len(val_lines)
    else:
        score = 0

    print(f"METRIC: {score}")


if __name__ == "__main__":
    main()
