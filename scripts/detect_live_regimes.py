#!/usr/bin/env python3
from __future__ import annotations

import sys

from trader_optimizer.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["detect-live-regimes", *sys.argv[1:]]))
