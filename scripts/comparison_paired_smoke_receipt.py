#!/usr/bin/env python3
"""Issue the immutable two-process Crazyflie comparison preflight receipt.

The two smoke reports must already exist and must have been produced
concurrently from the frozen source tree.  This CPU-only issuer revalidates
their source fingerprints, PPO update, memory evidence, and overlapping
steady-state samples before creating the one fixed receipt path.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import drone_run_matrix as matrix


def _entry(slot: int) -> dict[str, object]:
    path = matrix.COMPARISON_PAIRED_SMOKE_REPORTS[slot].resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Paired-smoke report is missing: {path}")
    return {
        "slot": slot,
        "seed": slot,
        "path": str(path.relative_to(matrix.ROOT)),
        "sha256": matrix.sha256_file(path),
    }


def issue_receipt() -> dict[str, object]:
    entries = [_entry(0), _entry(1)]
    reports = [
        matrix._validate_paired_smoke_report(entry, expected_slot=index)
        for index, entry in enumerate(entries)
    ]
    if len({report["source_set_sha256"] for report in reports}) != 1:
        raise ValueError("Paired-smoke reports use different source sets")
    matrix._paired_smoke_overlap_evidence(reports)
    body: dict[str, object] = {
        "schema_version": 1,
        "kind": matrix.COMPARISON_CONCURRENCY_RECEIPT_KIND,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "requested_max_concurrent_isaac_processes": 2,
        "effective_max_concurrent_isaac_processes": 2,
        "reports": entries,
    }
    receipt = {**body, "receipt_id": matrix.canonical_sha256(body)}
    destination = matrix.COMPARISON_CONCURRENCY_RECEIPT.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite immutable receipt: {destination}")
    encoded = (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination, follow_symlinks=False)
        directory_fd = os.open(
            destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    decision = matrix._validate_comparison_concurrency_receipt(
        matrix._comparison_concurrency_preflight_contract()
    )
    return {**decision, "receipt_output": str(destination)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        result = issue_receipt()
    except (FileExistsError, FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
