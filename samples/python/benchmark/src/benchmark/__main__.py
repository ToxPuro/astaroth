#!/bin/env python3

# Copyright (C) 2014-2026, Johannes Pekkila, Miikka Vaisala, Ondřej Míchal.
#
# This file is part of Astaroth.
#
# Astaroth is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Astaroth is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import pathlib
import sys

import benchmark.astaroth as ac
from benchmark import TestType, run_benchmark


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser("benchmark")

    parser.add_argument(
        "-c", "--config", type=pathlib.Path, default=ac.get_default_config()
    )
    parser.add_argument("dimensions", nargs=3, type=int)
    parser.add_argument(
        "-t",
        "--type",
        required=False,
        type=TestType,
        choices=list(TestType),
        default=TestType.STRONG_SCALING,
    )
    parser.add_argument(
        "--verify",
        required=False,
        action="store_true",
    )

    args: argparse.Namespace = parser.parse_args()

    args.config = args.config.absolute()

    if run_benchmark(args.config, args.type, args.dimensions, args.verify):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
