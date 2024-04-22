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


// Tester for boundary condition with GPU vs 
// TODO: Write first for single GPU and then for MPI. 
// TODO: Do first without taskgrap. Then add taskgraph.

// Loop this for each individual boundary condition in the list
//
// 1. Initialize a random grid on CPU.
//
// 2. Send grid to Device
//
// 3. Calculate boundary condition on Host and Device
//
// 4. Trasfer data from the device to host
//
// 5. Compare aqrray values close to the inner 
