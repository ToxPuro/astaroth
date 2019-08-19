#!/bin/bash
#!/bin/bash
if [ -z $AC_HOME ]
then
       echo "ASTAROTH_HOME environment variable not set, run \"source ./sourceme.sh\" in Astaroth home directory"
       exit 1
fi

KERNEL_DIR=${AC_HOME}"/src/core/kernels"
ACC_DIR=${AC_HOME}"/acc"
ACC_DEFAULT_SAS="mhd_solver/stencil_assembly.sas"
ACC_DEFAULT_SPS="mhd_solver/stencil_process.sps"
ACC_DEFAULT_HEADER="mhd_solver/stencil_definition.sdh"
ACC_DEFAULT_INCLUDE_DIR="mhd_solver"

${ACC_DIR}/clean.sh
${ACC_DIR}/build_acc.sh

ACC_SAS=${ACC_DEFAULT_SAS}
ACC_SPS=${ACC_DEFAULT_SPS}
ACC_HEADER=${ACC_DEFAULT_HEADER}
ACC_INCLUDE=${ACC_DEFAULT_INCLUDE_DIR}

while [ "$#" -gt 0 ]
do
	case $1 in
		-h|--help)
			echo "You can set a custom files for DSL under the path $AC_HOME/"
			echo "Example:"
			echo "compile_acc.sh -a custom_setup/custom_assembly.sas -p custom_setup/custom_process.sps --header custom_setup/custom_header.h"
			exit 0
			;;
		--header)
			shift
                        ACC_HEADER=${1}
			shift
                        echo "CUSTOM Header file!"
			;;
		-a|--assembly)
			shift
                        ACC_SAS=${1}
			shift
                        echo "CUSTOM Assembly file!"
			;;
		-p|--process)
			shift
                        ACC_SPS=${1}
			shift
			echo "CUSTOM Process file!"
			;;
		*)
			break
	esac
done

echo "Header file:" ${ACC_DIR}/${ACC_HEADER}
echo "Assembly file: ${ACC_DIR}/${ACC_SAS}"
echo "Process file: ${ACC_DIR}/${ACC_SPS}"

cd ${ACC_DIR}/${ACC_INCLUDE_DIR}
${ACC_DIR}/compile.sh ${ACC_DIR}/${ACC_SAS}
${ACC_DIR}/compile.sh ${ACC_DIR}/${ACC_SPS}
${ACC_DIR}/compile.sh ${ACC_DIR}/${ACC_HEADER}

#mv ${ACC_SAS} ${AC_HOME}/src/core/kernels
#mv ${ACC_SPS} ${AC_HOME}/src/core/kernels
#mv ${ACC_HEADER} ${AC_HOME}/include

#echo "Linking: " ${ACC_DIR}/${ACC_HEADER} " -> " ${AC_HOME}/include/stencil_defines.h
#ln -sf ${ACC_DIR}/${ACC_HEADER} ${AC_HOME}/include/stencil_defines.h
