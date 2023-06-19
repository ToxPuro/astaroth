#   Copyright (C) 2014-2023, Johannes Pekkila, Miikka Vaisala.
#   
#   This file is part of Astaroth.
#   
#   Astaroth is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#   
#   Astaroth is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#   
#   You should have received a copy of the GNU General Public License
#   along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.


# Module file for Julia tools in Astaroth

module AstarothTools
export ReadACData

function ReadACData()
    println("Reading snapshot data data...")
 
    # /tiara/ara/data/mvaisala/202304_haatouken/astaroth/config/samples/haatouken/output-snapshots/
    dirpath = "/tiara/ara/data/mvaisala/202304_haatouken/astaroth/config/samples/haatouken/output-snapshots/"
    directory = readdir(dirpath)

    println(directory)
    println(directory[1])

    binfile = dirpath * directory[1]
    println(binfile)
    xdim = 128 
    ydim = 128
    zdim = 256
    filesize = xdim*ydim*zdim
    binary_data = Array{Float64}(undef, filesize, 1);
    #binary_data = read(binfile, Float64)
    read!(binfile, binary_data)

    println(size(binary_data))
    println(filesize)
    #println(binary_data)
    println(typeof(binary_data))
    binary_data = reshape(binary_data, (xdim, ydim, zdim))
    println(size(binary_data))

    return 0 
end

end 
