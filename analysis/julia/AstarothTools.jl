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

    mesh_info = Dict("variable name" => "variable_value")

    mesh_info_int  = Dict("variable name" => 1)
    mesh_info_real = Dict("variable name" => 1.0)

    open(dirpath*"../mesh_info.list") do fobj
        for line in eachline(fobj)
            #println(line)
            line_content = split(line)
            #println(line_content)
            type = line_content[1]
            varname = line_content[2]
            varvalue = line_content[3]
            println(type, " ", varname, " ", varvalue)
            if type == "real"
                println(parse(Float64, varvalue))
                varnumber = parse(Float64, varvalue)
                mesh_info_real[varname] = varnumber
            elseif type == "int"
                varnumber = parse(Int, varvalue)
                mesh_info_int[varname] = varnumber
            elseif type == "int3" #TODO FIGURE THIS OUT 
                varnumber = parse(Float64, varvalue)
                mesh_info_int[varname] = varnumber
            elseif type == "size_t"
                varnumber = parse(Float64, varvalue)
                mesh_info_int[varname] = varnumber
            else
                mesh_info_str[varname] = varvalue
            end
        end
    end

    println(mesh_info)
    println(mesh_info_int)
    println(mesh_info_real)

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
