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

function ReadFilePiece(my_dir, binfile, xdim_loc, ydim_loc, zdim_loc)
    filesize = xdim_loc*ydim_loc*zdim_loc
    array_piece = Array{Float64}(undef, filesize, 1);
    read!(binfile, array_piece)

    file_info = split(my_dir, ".")
    file_info = split(file_info[1], "-")
    ##println(file_info)
    iix   = file_info[3]
    iiy   = file_info[4]
    iiz   = file_info[5]
    nstep = file_info[6] 

    ##println(size(array_piece))
    ##println(filesize)
    ##println(typeof(array_piece))
    array_piece = reshape(array_piece, (xdim_loc, ydim_loc, zdim_loc))
    ##println(size(array_piece))

    return array_piece
end 

function ReadACData(dirpath)
    println("Reading snapshot data data...")
 
    directory = readdir(dirpath)

    # Describe for of the the lists 
    mesh_info_int  = Dict("variable name" => 1)
    mesh_info_int3  = Dict("variable name" => (1,1,1))
    mesh_info_real = Dict("variable name" => 1.0)

    # Get info from mesh_info.list
    ReadMeshInfo(dirpath*"../")

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

    xdim_loc = xdim 
    ydim_loc = ydim
    zdim_loc = zdim

    step_numbers = String[]
    field_names  = String[]

    #TODO: Parse directory properly 
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
        println(my_dir)
        println(field_name, " ", step_number)

        push!(step_numbers, step_number)
        push!(field_names, field_name)

    end

    println( xdim_loc, " ", ydim_loc, " ", zdim_loc)
    println(step_numbers)
    println(field_names)

    #TODO: Remove repeating elements from step_numbers and field_names

    #xdim_loc = 246 
    #ydim_loc = 128
    #zdim_loc = 128

    xdims_list = 1:xdim_loc:xdim
    ydims_list = 1:ydim_loc:ydim
    zdims_list = 1:zdim_loc:zdim

    #TODO: This now goes throu every single file, which is wrong. We need to
    #TODO: parse the directory better .
    for my_dir in directory

        binfile = dirpath * my_dir
        ###println(binfile)

        for ii in xdims_list
            for jj in ydims_list
                for kk in zdims_list
                    array_piece = ReadFilePiece(my_dir, binfile, xdim_loc, ydim_loc, zdim_loc)
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
