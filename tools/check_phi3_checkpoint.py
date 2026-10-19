# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check local Phi-3 config and tensor shape manifest JSON files."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.logger import logger
from xllm.python.models.phi3_checkpoint import expected_phi3_shapes, validate_phi3_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Local Phi-3 config JSON")
    parser.add_argument("--manifest", required=True, type=Path, help="Weight key -> {shape: [rows, cols]} JSON")
    parser.add_argument("--tp-size", type=int, default=1, help="Evenly divisible tensor-parallel size")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        dimensions = validate_phi3_manifest(config, manifest, tp_size=args.tp_size)
    except (OSError, ValueError) as error:
        logger.error(f"Phi-3 checkpoint validation failed: {error}")
        return 1
    report = {
        **asdict(dimensions),
        "head_dim": dimensions.head_dim,
        "tp_size": args.tp_size,
        "checked_weights": len(expected_phi3_shapes(dimensions)),
    }
    json.dump(report, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
