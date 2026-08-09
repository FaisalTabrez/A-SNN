"""Official Spiking Speech Commands download, binning, and caching."""

from __future__ import annotations

import gzip
import pathlib
import shutil

from .event_mnist import torch
from .shd_benchmark import (
    SHD_BASE_URL,
    SHDConfig,
    _download_bytes,
    _download_to,
    _md5,
    bin_shd_events,
)


SSC_FILES = ("ssc_train.h5.gz", "ssc_valid.h5.gz", "ssc_test.h5.gz")


def ensure_ssc_files(root: pathlib.Path, *, download: bool) -> dict[str, pathlib.Path]:
    """Return decompressed official SSC files with Zenke Lab MD5 checks."""

    resolved = {
        "train": root / "ssc_train.h5",
        "validation": root / "ssc_valid.h5",
        "test": root / "ssc_test.h5",
    }
    if all(path.exists() for path in resolved.values()):
        return resolved
    if not download:
        missing = [str(path) for path in resolved.values() if not path.exists()]
        raise FileNotFoundError("missing SSC files: " + ", ".join(missing))
    md5_text = _download_bytes(f"{SHD_BASE_URL}/md5sums.txt").decode("utf-8")
    hashes = {
        fields[1]: fields[0]
        for line in md5_text.splitlines()
        if len(fields := line.split()) == 2
    }
    root.mkdir(parents=True, exist_ok=True)
    for filename in SSC_FILES:
        gz_path = root / filename
        h5_path = root / filename.removesuffix(".gz")
        if h5_path.exists():
            continue
        if not gz_path.exists():
            print(f"Downloading {filename} from Zenke Lab...")
            _download_to(f"{SHD_BASE_URL}/{filename}", gz_path)
        expected = hashes.get(filename)
        if expected and _md5(gz_path) != expected:
            raise RuntimeError(f"MD5 verification failed for {gz_path}")
        temporary = h5_path.with_suffix(h5_path.suffix + ".part")
        with gzip.open(gz_path, "rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        temporary.replace(h5_path)
    return resolved


def load_ssc_tensors(
    config: SHDConfig,
    *,
    validation_samples: int = 0,
):
    """Load official SSC train/validation/test tensors with deterministic limits."""

    if torch is None:
        raise ImportError("SSC preprocessing requires PyTorch")
    if config.input_neurons != 700 or config.classes != 35:
        raise ValueError("SSC requires 700 input neurons and 35 classes")
    root = pathlib.Path(config.data_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    paths = ensure_ssc_files(root, download=config.download)
    train_events, train_labels = _load_or_build_ssc_split(
        paths["train"], root, "train", config
    )
    validation_events, validation_labels = _load_or_build_ssc_split(
        paths["validation"], root, "validation", config
    )
    test_events, test_labels = _load_or_build_ssc_split(
        paths["test"], root, "test", config
    )
    generator = torch.Generator(device="cpu").manual_seed(config.data_seed)
    train_events, train_labels = _limit_split(
        train_events, train_labels, config.train_samples, generator
    )
    validation_events, validation_labels = _limit_split(
        validation_events, validation_labels, validation_samples, generator
    )
    test_events, test_labels = _limit_split(
        test_events, test_labels, config.test_samples, generator
    )
    return (
        train_events,
        train_labels,
        validation_events,
        validation_labels,
        test_events,
        test_labels,
    )


def _load_or_build_ssc_split(
    h5_path: pathlib.Path,
    root: pathlib.Path,
    split: str,
    config: SHDConfig,
):
    duration_ms = round(config.duration_seconds * 1000)
    cache = root / (
        f"ssc_{split}_t{config.timesteps}_c{config.input_neurons}_"
        f"d{duration_ms}ms.pt"
    )
    if cache.exists():
        payload = torch.load(cache, map_location="cpu", weights_only=True)
        return payload["events"], payload["labels"]
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("SSC preprocessing requires h5py") from exc
    print(f"Binning {h5_path.name} into {config.timesteps} temporal steps...")
    with h5py.File(h5_path, "r") as handle:
        labels = torch.as_tensor(handle["labels"][:], dtype=torch.long)
        events = torch.zeros(
            (labels.shape[0], config.timesteps, config.input_neurons),
            dtype=torch.uint8,
        )
        times = handle["spikes"]["times"]
        units = handle["spikes"]["units"]
        for index in range(labels.shape[0]):
            events[index] = bin_shd_events(
                times[index],
                units[index],
                timesteps=config.timesteps,
                input_neurons=config.input_neurons,
                duration_seconds=config.duration_seconds,
            )
            if (index + 1) % 2000 == 0:
                print(f"  {split}: {index + 1}/{labels.shape[0]}")
    temporary = cache.with_suffix(cache.suffix + ".part")
    torch.save({"events": events, "labels": labels}, temporary)
    temporary.replace(cache)
    return events, labels


def _limit_split(events, labels, limit: int, generator):
    if limit <= 0 or limit >= events.shape[0]:
        return events, labels
    indices = torch.randperm(events.shape[0], generator=generator)[:limit]
    return events.index_select(0, indices), labels.index_select(0, indices)
