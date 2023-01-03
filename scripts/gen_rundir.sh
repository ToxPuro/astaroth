#!/bin/bash

print_usage(){
    echo "$0 [OPTIONS] [AC_config_param=foo ...]"
    echo ""
    echo "Options"
    echo "======="
    echo " --output <name of produced rundir>"
    echo ""
    echo " --stripe <num_stripes>"
    echo "   stripe the rundir with num_stripes stripes"
    echo ""
    echo "Astaroth configuration"
    echo "----------------------"
    echo " --ac_run_mpi <filepath>"
    echo "   path to ac_run_mpi binary that will be called in the SLURM batch script"
    echo ""
    echo " --config <filepath>"
    echo "   path to astartoh conf that will be copied to the rundir"
    echo ""
    echo "   Using config overrides, you can change values from the reference astaroth.conf file, e.g.:"
    echo "     gen_rundir --ac_run_mpi=/path/to/ac_run_mpi --config=/path/to/astaroth.conf AC_eta=5e-3 AC_nu_visc=2.5e-5"
    echo ""
    echo " --varfile <filepath>"
    echo "   path to varfile to use as initial mesh for run"
    echo ""
    #echo " --kernel <name>"
    echo " --dims <x,y,z>"
    echo "   override dims in configuration file, and calculate dsx, dsy, dsz."
    echo "   DEACTIVATED BECAUSE DSX, DSY, DSZ must be set in the source code"
    echo ""
    echo "Launcher configuration"
    echo "----------------------"
    echo " --render [rendering option]"
    echo ""
    echo "  Rendering options are:"
    echo "  --render off"
    echo "    don't render, just call sbatch"
    echo "  --render deferred"
    echo "    render slices once, after simulation has run"
    echo ""
    echo "SLURM configuration"
    echo "----------------------"
    echo "    --timelimit <SLURM time limit>"
    echo "    --num-procs <num procs>"
    echo "    --account <SLURM account> --partition <SLURM partition>"
    echo "    [AC_foo=X AC_bar=Y AC_bax=123]"
}

script_dir=$(realpath $(dirname "${BASH_SOURCE[0]}"))
#WIP
#render_companion_batch=$(realpath ${script_dir}/../analysis/viz_tools/on-the-fly-render)

config=$(realpath ${script_dir}/../config/astaroth.conf)
output_dir=astaroth_rundir
dims=""
ac_run_mpi_binary=$(realpath ${script_dir}/../build/ac_run_mpi)
render=deferred
timelimit=00:15:00
num_procs=8
account=project_462000120
partition=small-g

if [[ -n "$AC_CONFIG" ]]; then
    config="$AC_CONFIG"
fi

if [[ -n "$AC_RUN_MPI" ]]; then
    ac_run_mpi_binary="$AC_RUN_MPI"
fi

if [[ -n "$AC_VARFILE" ]]; then
    varfile="$AC_VARFILE"
fi

OPTS=`getopt -o c:o:d:a:r:t:n:v:A:p:s:kh -l config:,output:,dims:,ac_run_mpi:,render:,timelimit:,num-procs:,varfile:,account:,partition:,stripe:,kernel,help -n "$0"  -- "$@"`
if [ $? != 0 ] ; then echo "Failed to parse args" >&2 ; exit 1 ; fi

eval set -- "$OPTS"

init_mesh=kernel
while true;do
case "$1" in
-h|--help)
        print_usage
        exit 0
        ;;
-c|--config)
	shift
	config="$1"
	;;
-o|--output)
        shift
        output_dir="$1"
        ;;
-s|--stripe)
	shift
	stripes="$1"
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
	ac_run_mpi_binary="$1"
	;;
-r|--render)
	shift
	render="$1"
	;;
-t|--timelimit)
        shift
	timelimit="$1"
	;;
-n|--num-procs)
	shift
	num_procs=$1
	;;
-v|--varfile)
        shift
	init_mesh=varfile
	varfile="$1"
	;;
-k|--kernel)
	shift
	init_mesh=kernel
	#kernel=$1
	;;
-A|--account)
        shift
	account="$1"
	;;
-p|--partition)
	shift
	partition="$1"
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
    slice_renderer=$(realpath ${script_dir}/../analysis/viz_tools/render_slices)
    launcher="sbatch --parsable"
    ;;
off)
    launcher="sbatch --parsable"
    ;; 
*)
    echo "ERROR: can't recognize render argument \"$render\". Options are \"off\" and \"deferred\""
    exit 1
    ;;
esac
    cat > "$output_dir/submit.sh" << EOF | grep -v '^[[:blank:]]*$'
#!/bin/bash
rundir=\$(dirname "\${BASH_SOURCE[0]}")
if [[ "\$(realpath \$rundir)" != "\$(realpath \$PWD)" ]]; then
    echo "Error: please call ./submit.sh from the rundir, not from elsewhere"
    exit 1
fi
SLURM_SIM_JOB=\$($launcher simulation.sbatch)
if [[ \$? -eq 0 ]]; then
    echo "Submitted simulation slurm job \$SLURM_SIM_JOB"
$(if [[ "$render" == "deferred" ]];then
    echo "    echo \"Queueing postprocessing to start after simulation\""
    echo "    SLURM_POSTPROCESS_JOB=\$(sbatch --dependency=afterany:\$SLURM_SIM_JOB postprocess.sbatch)"
    echo "    echo \"Submitted postprocessing slurm job \$SLURM_POSTPROCESS_JOB\""
fi)

    echo "to follow the simulation, you can try the following once it has started:"
    echo "  tail -F slurm-simulation*.out"
    echo ""
    echo "to follow all your current SLURM jobs:"
    echo "  watch \"squeue --me\" "
else
    echo "\$SLURM_SIM_JOB"
fi
EOF

    chmod +x "$output_dir/submit.sh"
}


gen_simulation_sbatch(){
    num_gpus=$num_procs
    if [[ $num_gpus -gt 8 ]]; then
        num_gpus=8
    fi

    #Hardcoded for now, but these could be queried from SLURM
    gpus_per_node=8
    num_nodes=$((x=num_procs+gpus_per_node-1, x/gpus_per_node))
 
    ac_run_mpi_args=""

    case "$init_mesh" in
    varfile)
      ac_run_mpi_args="--from-pc-varfile $(realpath $varfile)"
      ;;
    kernel)
      ac_run_mpi_args="--kernel"
      ;;
    esac
    
    ac_run_mpi_realpath=$(realpath $ac_run_mpi_binary)

    cat > $output_dir/simulation.sbatch << EOF | grep -v '^[[:blank:]]*$'
#!/bin/bash
#SBATCH --account=$account
#SBATCH --partition=$partition
#SBATCH --gres=gpu:$num_gpus
#SBATCH --ntasks=$num_procs
#SBATCH --nodes=$num_nodes
#SBATCH --time=$timelimit
#SBATCH --output=slurm-simulation-%j.out

module purge

module load CrayEnv
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm
module load buildtools
module load cray-python

if [[ ! -f astaroth.conf ]]; then
    echo "astaroth.conf does not exist or is not a regular file"
    exit 1
fi

if [[ ! -x "$ac_run_mpi_realpath" ]]; then
    echo "$ac_run_mpi_realpath does not exist or is not executable"
    exit 1
fi

srun $ac_run_mpi_realpath --config astaroth.conf $ac_run_mpi_args
EOF
}

gen_postprocess_sbatch(){
    cat > $output_dir/postprocess.sbatch << EOF | grep -v '^[[:blank:]]*$'
#!/bin/bash
#SBATCH --account=$account
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --time=00:15:00
#SBATCH --output=slurm-postprocess-%j.out

if [[ ! -d output-slices ]]; then
    echo "Expected data in output-slices, but output-slices is not a directory"
    exit 1
fi
srun $slice_renderer --input output-slices/*/*
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
    )" $config > "$output_dir/astaroth.conf"
}

if [[ -e "$output_dir" ]]; then
    suffix=1
    stem="$output_dir"
    while true; do
        output_dir="$stem.$suffix"
        if [ ! -e "$output_dir" ]; then
	    break
	fi
	suffix=$((suffix+1))
    done
fi

mkdir -p "$output_dir"

#Show the user what they are generating
grid_size=$((AC_nx * AC_ny * AC_nz))
work_per_proc=$((grid_size / num_procs))
echo "Generating rundir at $output_dir"
echo ""
echo "Run dimensions"
echo "--------------"
echo "     num procs: $num_procs"
echo " grid dims are: $AC_nx,$AC_ny,$AC_nz"
echo "    local grid: $work_per_proc cells"
echo ""
echo "Astaroth configuration"
echo "----------------------"
echo " ac_run_mpi is:"
echo "  $(realpath $ac_run_mpi_binary)"
echo " mesh will be initialized from a $init_mesh: "
case "$init_mesh" in
kernel)
    #TODO: this is hardcoded here and in the source code, change when that changes
    echo "  randomize"
    ;;
varfile)
    echo "  $(realpath $varfile)"
    ;;
esac
echo " config file will be copied from:"
echo "  $config"
echo " config overrides are:"
for key in "${!params[@]}"; do
    echo "$key = ${params[$key]}"
done
echo ""
echo "SLURM params"
echo "------------"
echo "  partition: $partition"
echo "    account: $account"
echo "  num procs: $num_procs"
echo " time limit: $timelimit"
echo ""
echo " slice rendering is $render"

echo ""

gen_submit_sh
gen_simulation_sbatch
case "$render" in
deferred)
    gen_postprocess_sbatch
    ;;
esac
gen_astaroth_conf

if [[ -n "$stripes" ]]; then
    if type lfs; then 
        echo "Striping rundir into $stripes stripes"
        lfs setstripe --stripe-count $stripes "$output_dir"
    else
	echo "Can't find lfs, not striping"
    fi
else
    echo "Not striping rundir"
fi

echo ""
echo "Finished generating, to run the simulation, do:"
echo "  cd $output_dir && ./submit.sh"
echo ""
