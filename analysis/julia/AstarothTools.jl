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

    # Describe for of the the lists 
    mesh_info_int  = Dict("variable name" => 1)
    mesh_info_int3  = Dict("variable name" => (1,1,1))
    mesh_info_real = Dict("variable name" => 1.0)

    # Read lines in the file 
    open(dirpath*"../mesh_info.list") do fobj
        for line in eachline(fobj)
            println(line)
            line_content = split(line)
            #println(line_content)
            type = line_content[1]
            varname = line_content[2]
            varvalue = line_content[3]
            #println(type, " ", varname, " ", varvalue)
            if type == "real"
                varnumber = parse(Float64, varvalue)
                mesh_info_real[varname] = varnumber
            elseif type == "int" 
                varnumber = parse(Int, varvalue)
                mesh_info_int[varname] = varnumber
            elseif type == "int3"
                varnumber3 = (parse(Int, line_content[3]), parse(Int, line_content[4]), parse(Int, line_content[5]))
                mesh_info_int3[varname] = varnumber3 
            elseif type == "size_t"
                varnumber = parse(Float64, varvalue)
                mesh_info_int[varname] = varnumber
            else
                mesh_info_str[varname] = varvalue
            end
        end
    end
    delete!(mesh_info_int, "variable name")
    delete!(mesh_info_int3, "variable name")
    delete!(mesh_info_real, "variable name")

    println(mesh_info_int)
    println(mesh_info_int3)
    println(mesh_info_real)

    println(directory)

    xdim = 256
    ydim = 256
    zdim = 256

    arraydims = (xdim, ydim, zdim)

    println(arraydims)
    whole_array = zeros(arraydims)

    for my_dir in directory

        binfile = dirpath * my_dir
        println(binfile)
        xdim_loc = 128 
        ydim_loc = 128
        zdim_loc = 256
        filesize = xdim_loc*ydim_loc*zdim_loc
        binary_data = Array{Float64}(undef, filesize, 1);
        read!(binfile, binary_data)

        file_info = split(my_dir, ".")
        file_info = split(file_info[1], "-")
        println(file_info)
        iix   = file_info[3]
        iiy   = file_info[4]
        iiz   = file_info[5]
        nstep = file_info[6] 

        println(size(binary_data))
        println(filesize)
        println(typeof(binary_data))
        binary_data = reshape(binary_data, (xdim_loc, ydim_loc, zdim_loc))
        println(size(binary_data))
        whole_array[1:128, 1:128, 1:256] = binary_data
    end 

    return 0 
end

end 
