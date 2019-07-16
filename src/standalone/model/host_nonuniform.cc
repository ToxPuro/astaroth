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
#include "host_forcing.h"

#include "core/math_utils.h"

// Describes the grid geometry. Should be monotonic function. 
// TODO: Add alternatives.
AcReal
grid_function(const AcReal a_grid, const AcReal zeta, const int der_degree)
{

    //Using now sinh() as an example.
    if (der_degree == 0) {
        return sinh(zeta);
    } else if (der_degree == 1) {
        return a_grid*cosh(zeta);
    } else if (der_degree == 2) {
        return (a_grid*a_grid)*sinh(zeta);
    }

}

// Calculates the grid geometry function (Pencil Code manual section 5.4)
AcReal 
grid_geometry(const AcReal zeta, const AcReal zeta_star, const AcReal z0,
              const AcReal AxisLength, const AcReal a_grid, const int Ngrid)
{

    AcReal denominator;
    denominator = grid_function(a_grid, (zeta - zeta_star), 0) 
                + grid_function(a_grid, (zeta_star - int(0)), 0)
    
    AcReal nominator;
    nominator   = grid_function(a_grid, (Ngrid - int(1) - zeta_star), 0) 
                + grid_function(a_grid, (zeta_star - int(0)), 0)

    return z0 + AxisLength*(nominator/denominator);

}
mesh_info.real_params[AC_ny]

// Calculated the grid geometry function for all directions
AcReal3
grid_geometry_xyz(const AcReal3 zeta)
{
    return (Vector){
    grid_geometry_xyz(zeta.x, zeta_star.x, mesh_info.real_params[AC_xorig], 
                      mesh_info.real_params[AC_xlen], a_grid, mesh_info.int_params[AC_nx]),

    grid_geometry_xyz(zeta.y, zeta_star.y, mesh_info.real_params[AC_yorig], 
                      mesh_info.real_params[AC_ylen], a_grid, mesh_info.int_params[AC_ny]),

    grid_geometry_xyz(zeta.z, zeta_star.z, mesh_info.real_params[AC_zorig], 
                      mesh_info.real_params[AC_zlen], a_grid, mesh_info.int_params[AC_nz])
    };
}

//TODO: 

//Solve for zeta_star 

//Loads parameters to GPU 






