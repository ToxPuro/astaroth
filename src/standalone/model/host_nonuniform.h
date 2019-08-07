
/*
    Copyright (C) 2014-2019, Johannes Pekkilae, Miikka Vaeisalae.

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

/**
 * @file
 * \brief Brief info.
 *
 * Detailed info.
 *
 */
#pragma once
#include "astaroth.h"

AcReal
grid_function(const AcReal a_grid, const AcReal zeta, const int der_degree);

AcReal 
grid_geometry(const AcReal zeta, const AcReal zeta_star, const AcReal z0,
              const AcReal AxisLength, const AcReal a_grid, const int Ngrid);

AcReal3
solve_zeta_star();

AcReal3
grid_geometry_xyz(const AcReal3 zeta, const AcMeshInfo& mesh_info);



