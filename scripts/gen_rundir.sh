#!/bin/bash

print_usage(){
    echo "$0  --output <output_dir>"
    echo "    --dims <x,y,z>"
    echo "    --config <reference conf file> --ac_run_mpi <path to ac_run_mpi binary> --varfile <PC var file path>"
    echo "    --render=<off | deferred >"
    echo "    --timelimit <SLURM time limit> --num-procs <num procs>"
    echo "    --account <SLURM account> --partition <SLURM partition>"
    echo "    [AC_foo=X AC_bar=Y AC_bax=123]"
}

script_dir=$(realpath $(dirname "${BASH_SOURCE[0]}"))
render_companion_batch=$(realpath ${script_dir}/../analysis/viz_tools/deferred-render)

config=$(realpath ${script_dir}/../config/astaroth.conf)
output_dir=astaroth_rundir
dims=""
ac_run_mpi_binary=$(realpath ${script_dir}/../build/ac_run_mpi)
render=deferred
timelimit=00:15:00
num_procs=8
account=project_462000120
partition=pilot

if [[ -n "$AC_CONFIG" ]]; then
    config="$AC_CONFIG"
fi

if [[ -n "$AC_RUN_MPI" ]]; then
    ac_run_mpi_binary="$AC_RUN_MPI"
fi

if [[ -n "$AC_VARFILE" ]]; then
    varfile="$AC_VARFILE"
fi

OPTS=`getopt -o c:o:d:a:r:t:n:v:A:p:h -l config:,output:,dims:,ac_run_mpi:,render:,timelimit:,num-procs:,varfile:,account:,partition:,help -n "$0"  -- "$@"`
if [ $? != 0 ] ; then echo "Failed to parse args" >&2 ; exit 1 ; fi

eval set -- "$OPTS"

while true;do
case "$1" in
-h|--help)
        print_usage
        exit 0
        ;;
-c|--config)
	shift
	config=$1
	;;
-o|--output)
        shift
        output_dir=$1
        ;;
-d|--dims)
        shift
	OLDIFS=$IFS
	IFS=','
	read -a dims <<< "$1"
	IFS=$OLDIFS
        ;;
-a|--ac_run_mpi)
	shift
	ac_run_mpi_binary=$1
	;;
-r|--render)
	shift
	render=$1
	;;
-t|--timelimit)
        shift
	timelimit=$1
	;;
-n|--num-procs)
	shift
	num_procs=$1
	;;
-v|--varfile)
        shift
	varfile=$1
	;;
-A|--account)
        shift
	account=$1
	;;
-p|--partition)
	shift
	partition=$1
	;;
--)
        shift
        break
        ;;
*)
        break
        ;;
esac
shift
done

declare -A params
OLDIFS=$IFS
IFS='='
for param in "$@"; do
    read key val <<< $param
    if [[ -z "$val" ]]; then
        echo "parameter $key does not name a value, give config parameters like so: AC_foo=bar"
    fi
    params["$key"]="$val"
done
IFS=$OLDIFS

if [[ ! -f "$config" ]]; then
    echo "ERROR: astaroth config \"$config\" is not a file"
    exit 1
fi

if [[ ! -x "$ac_run_mpi_binary" ]]; then
    echo "ERROR: astaroth binary \"$ac_run_mpi_binary\" is not an executable"
    exit 1
fi

if [[ ! -f "$varfile" ]]; then
    echo "ERROR: varfile \"$varfile\" is not an executable"
    exit 1
fi

get_config_param(){
    param_name=$1
    awk "/$param_name/ {print \$3}" < $config
}

if [[ -z "$dims" ]]; then
    AC_nx=$(get_config_param AC_nx)
    AC_ny=$(get_config_param AC_ny)
    AC_nz=$(get_config_param AC_nz)
else
    if [[ ${#dims[@]} -ne 3 ]]; then
	echo "ERROR: dims must be three values separated by commas: x,y,z"
        exit 1
    fi
    AC_nx=${dims[0]}
    AC_ny=${dims[1]}
    AC_nz=${dims[2]}
    echo "WARNING: dims will not be used"
fi


#Calculate dsx, dsy, dsz
#HMMM: this may produce a slightly different value from what's in the binary...
#AC_dsx=$(bc -l <<< "6.2831853070 / $AC_nx")
#AC_dsy=$(bc -l <<< "6.2831853070 / $AC_ny")
#AC_dsz=$(bc -l <<< "6.2831853070 / $AC_nz")

#TODO: options for collective and distributed

gen_submit_sh(){
case "$render" in
deferred)
    launcher="$(realpath $render_companion_batch)"
    ;;
off)
    launcher=sbatch
    ;; 
*)
    echo "ERROR: can't recognize render argument \"$render\". Options are \"off\" and \"deferred\""
    exit 1
    ;;
esac
    cat > $output_dir/submit.sh << EOF | grep -v '^[[:blank:]]*$'
#!/bin/bash
rundir=\$(dirname "\${BASH_SOURCE[0]}")
if [[ "\$(realpath \$rundir)" != "\$(realpath \$PWD)" ]]; then
    echo "Please change dir to the directory first"
fi
$launcher simulation.sbatch
EOF

    chmod +x $output_dir/submit.sh
}

gen_simulation_sbatch(){
    num_gpus=$num_procs
    if [[ $num_gpus -gt 8 ]]; then
        num_gpus=8
    fi

    #Hardcoded for now, but these could be queried from SLURM
    gpus_per_node=8
    num_nodes=$((x=num_procs+gpus_per_node-1, x/gpus_per_node))
    
    cat > $output_dir/simulation.sbatch << EOF | grep -v '^[[:blank:]]*$'
#!/bin/bash
#SBATCH --account=$account
#SBATCH --partition=$partition
#SBATCH --gres=gpu:$num_gpus
#SBATCH --ntasks=$num_procs
#SBATCH --nodes=$num_nodes
#SBATCH --time=$timelimit

module purge

module load CrayEnv
module load PrGEnv-cray
module load craype-accel-amd-gfx90a
module load rocm
module load buildtools
module load cray-python

srun $(realpath $ac_run_mpi_binary) --config astaroth.conf --from-pc-varfile $(realpath $varfile)
EOF
}

gen_astaroth_conf(){
    awk "$(
cat << EOF
        $(for key in "${!params[@]}"; do
	    echo "/$key/{printf \"$key = ${params[$key]}\n\"; next}";
        done )
    {print}
EOF
    )" $config > $output_dir/astaroth.conf
}

if [[ -e $output_dir ]]; then
    suffix=1
    stem=$output_dir
    while true; do
        output_dir=$stem.$suffix
        if [ ! -e $output_dir ]; then
	    break
	fi
	suffix=$((suffix+1))
    done
fi

mkdir -p $output_dir

grid_size=$((AC_nx * AC_ny * AC_nz))
work_per_proc=$((grid_size / num_procs))
echo "Generating rundir at $output_dir"
echo "Generating sbatch script with $num_procs procs and a grid of $AC_nx,$AC_ny,$AC_nz"
echo "Local grid per GPU has $work_per_proc cells"


echo "Generating submit.sh"
gen_submit_sh
echo "Generating simulation.sbatch"
gen_simulation_sbatch
echo "Generating astaroth.conf"
gen_astaroth_conf

echo "Done generation, to run the simulation, do:"
echo "  cd $output_dir"
echo "  ./submit.sh"
