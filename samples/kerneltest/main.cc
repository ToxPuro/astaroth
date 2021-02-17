/*
    Copyright (C) 2014-2020, Johannes Pekkila, Miikka Vaisala.

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
#include "astaroth.h"
#include "astaroth_utils.h"

int
main(void)
{
    const int nn = 64;

    AcMeshInfo info;
    info.int_params[AC_nx] = info.int_params[AC_ny] = info.int_params[AC_nz] = nn;
    acHostUpdateBuiltinParams(&info);

    Device device;
    acDeviceCreate(0, info, &device);
    acDevicePrintInfo(device);

    const int3 start = (int3){NGHOST, NGHOST, NGHOST};
    const int3 end   = (int3){NGHOST + nn, NGHOST + nn, NGHOST + nn};

    acDevice_solve(device, STREAM_DEFAULT, start, end);

    acDeviceDestroy(device);
}
