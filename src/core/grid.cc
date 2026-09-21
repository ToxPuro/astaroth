/*
    Copyright (C) 2020-2026, Johannes Pekkilä, Miikka Väisälä, Oskar Lappi, Touko Puro

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

#include "astaroth_grid_private.h"

extern Grid grid;

bool
acGridInitialized()
{
#if AC_RUNTIME_COMPILATION
    return false;
#else
    return grid.initialized;
#endif
}
