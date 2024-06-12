/*
    Copyright (C) 2014-2024, Johannes Pekkila, Miikka Vaisala.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once
#include <stdint.h> //uint64_t

#include "errchk.h"
#include "math_utils.h" // uint3_64

uint3_64 decompose(const uint64_t target);

int getPid(const int3 pid_raw, const uint3_64 decomp);

int3 getPid3D(const uint64_t pid, const uint3_64 decomp);

/** Assumes that contiguous pids are on the same node and there is one process per GPU. */
bool onTheSameNode(const uint64_t pid_a, const uint64_t pid_b);
