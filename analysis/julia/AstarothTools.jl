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

function ReadMeshInfo(info_path)
    # Get setup information from mesh_info.list file

    # Describe for of the the lists 
    mesh_info_int  = Dict("variable name" => 1)
    mesh_info_int3  = Dict("variable name" => (1,1,1))
    mesh_info_real = Dict("variable name" => 1.0)

    # Read lines in the file 
    open(info_path*"mesh_info.list") do fobj
        for line in eachline(fobj)
            ##println(line)
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
                varnumber3 = (parse(Int, line_content[3]), 
                              parse(Int, line_content[4]), 
                              parse(Int, line_content[5]))
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

    return mesh_info_int, mesh_info_int3, mesh_info_real

end 

function ReadFilePiece(binfile, xdim_loc, ydim_loc, zdim_loc)
    filesize = xdim_loc*ydim_loc*zdim_loc
    array_piece = Array{Float64}(undef, filesize, 1);
    read!(binfile, array_piece)

    ##println(size(array_piece))
    ##println(filesize)
    ##println(typeof(array_piece))
    array_piece = reshape(array_piece, (xdim_loc, ydim_loc, zdim_loc))
    ##println(size(array_piece))

    return array_piece
end 

function RemoveRepetion(list_names, name_now)
    repeat = false
    for name in list_names
        if name == name_now
            repeat = true
        end
    end
    if repeat == false 
        push!(list_names, name_now)
    end

    return list_names
end

function ParseDataDirectory(arraydims, dirpath)
    directory = readdir(dirpath)

    xdim_loc = arraydims[1]
    ydim_loc = arraydims[2]
    zdim_loc = arraydims[3]

    step_numbers = String[]
    field_names  = String[]

    for my_dir in directory
        my_dir = split(my_dir, ".")
        my_dir = split(my_dir[1], "-")
        field_name = my_dir[1]
        step_number = my_dir[6]
        if parse(Int, my_dir[3]) < xdim_loc && parse(Int, my_dir[3]) > 0 
            xdim_loc = parse(Int, my_dir[3])
        end
        if parse(Int, my_dir[4]) < ydim_loc && parse(Int, my_dir[4]) > 0 
            ydim_loc = parse(Int, my_dir[4])
        end
        if parse(Int, my_dir[5]) < zdim_loc && parse(Int, my_dir[5]) > 0 
            zdim_loc = parse(Int, my_dir[5])
        end
        #println(my_dir)
        #println(field_name, " ", step_number)

        step_numbers = RemoveRepetion(step_numbers, step_number)
        
        field_names = RemoveRepetion(field_names, field_name)

    end

    return xdim_loc, ydim_loc, zdim_loc, step_numbers, field_names
end

function FetchStepNumbers(dirpath)
    mesh_info_int, mesh_info_int3, mesh_info_real = ReadMeshInfo(dirpath*"../")

    arraydims = (mesh_info_int["AC_nx"], 
                 mesh_info_int["AC_ny"], 
                 mesh_info_int["AC_nz"])

    xdim_loc, ydim_loc, zdim_loc, step_numbers, field_names = ParseDataDirectory(arraydims, dirpath)
   
    return step_numbers
end

#TODO: Set varios data field into datatype elements and output the datatype
#TODO: from the function
function ReadACData(dirpath, step)
    println("Reading snapshot data data...")
 

    # Get info from mesh_info.list
    mesh_info_int, mesh_info_int3, mesh_info_real = ReadMeshInfo(dirpath*"../")

    arraydims = (mesh_info_int["AC_nx"], 
                 mesh_info_int["AC_ny"], 
                 mesh_info_int["AC_nz"])

    println(arraydims)
    whole_array = zeros(arraydims)

    xdim_loc, ydim_loc, zdim_loc, step_numbers, field_names = ParseDataDirectory(arraydims, dirpath)

    xdims_list = 1:xdim_loc:arraydims[1]
    ydims_list = 1:ydim_loc:arraydims[2]
    zdims_list = 1:zdim_loc:arraydims[3]

    step = string(step)

    for field in field_names
        for ii in xdims_list
            for jj in ydims_list
                for kk in zdims_list
                    filename = field*"-segment-"*string(ii-1)*
                               "-"*string(jj-1)*"-"*string(kk-1)*
                               "-"*step*".mesh"
                    binfile = dirpath*filename
                    println(filename)
                    array_piece = ReadFilePiece(binfile, xdim_loc, ydim_loc, zdim_loc)
                    whole_array[ii:(ii+xdim_loc-1), 
                                jj:(jj+ydim_loc-1), 
                                kk:(kk+zdim_loc-1)] = array_piece
                end
            end
        end
    end 

    return 0 
end

end 
