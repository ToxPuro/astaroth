#!/bin/bash

#defaults
default_num_procs=8
default_num_nodes=2
default_num_gpus=4
default_partition=gpu
sbatch_settings=""

partition=$default_partition

script_name=$0

print_usage(){
    echo "Usage: $script_name -n <num_procs> [Options]"
    echo "      Runs the ./benchmark sample, which will write benchmark results to a csv file"
    echo "      Remember to run this script from your build directory"
    echo "      The benchmarks are submitted with sbatch, unless the -i option is passed"
    echo "Parameters:"
    echo "      -n <num_procs>"
    echo "              number of tasks for slurm"
    echo "      -x"
    echo "              don't use misconfigured nodes"
    echo "      -P"
    echo "              Use pinned settings"
    echo "Options:"
    echo "      -p <partition>"
    echo "              which partition to use for slurm, default=$default_partition"
    echo "      -h"
    echo "              Print this message"
}

while getopts :n:t:p:hxP opt
do
    case "$opt" in
        n)
            num_procs=$OPTARG
            num_nodes=$(( 1 + ($num_procs - 1)/4))
	    if [[ $((num_procs > 4 )) ]]
	    then
		num_gpus=4
		echo "Setting gpu to 4"
	    else
		num_gpus=$num_procs
            fi
        ;;
	P)
	    pinned=1
	;;
	x)
	    x_misconfigured=1
	;;
        p)
            partition=$OPTARG
        ;;
        h)
            print_usage
            exit 0
        ;;
        ?)
            print_usage
            exit 1
    esac
done

prefix=benchmark_meshsize

for dir in ${prefix}_*
do

if [[ $dir =~ ^.*1792$ ]]; then
	dim=1792
elif [[ $dir =~ ^.*1024$ ]]; then
	dim=1024
elif [[ $dir =~ ^.*512$ ]]; then
	dim=512
elif [[ $dir =~ ^.*448$ ]]; then
	dim=448
elif [[ $dir =~ ^.*256$ ]]; then
	dim=256
elif [[ $dir =~ ^.*128$ ]]; then
	dim=128
else
	dim=256
fi

echo "Submitting sbatch $dir/.benchmark for ${num_procs} procs, ${num_nodes} nodes, ${num_gpus} gpus, mesh size ${dim}^3 to partition ${partition}"

if [ -z "$x_misconfigured" ]
then
echo "RUNNING ON ALL NODES"
sbatch << EOF
#!/bin/sh
#SBATCH --job-name=astaroth
#SBATCH --account=project_2000403
#SBATCH --time=01:00:00
#SBATCH --mem=76000
#SBATCH --partition=${partition}
#SBATCH --output=benchmark-1-%j.out
#SBATCH --gres=gpu:v100:${num_gpus}
#SBATCH --ntasks-per-socket=2
#SBATCH -n ${num_procs}
#SBATCH -N ${num_nodes}
module load gcc/8.3.0 cuda/10.1.168 cmake openmpi/4.0.3-cuda nccl
cd $dir && srun ./benchmark $dim $dim $dim && cd ..
EOF
else
echo "NOT RUNNING ON BAD NODES"
sbatch << EOF
#!/bin/sh
#SBATCH --job-name=astaroth
#SBATCH --account=project_2000403
#SBATCH --time=01:00:00
#SBATCH --mem=76000
#SBATCH --partition=${partition}
#SBATCH --output=benchmark-1-%j.out
#SBATCH --gres=gpu:v100:${num_gpus}
#SBATCH -x r04g[05-06],r02g02,r14g04,r04g07,r16g07,r18g[02-03],r15g08,r17g06,r13g04
#SBATCH --ntasks-per-socket=2
#SBATCH -n ${num_procs}
#SBATCH -N ${num_nodes}
module load gcc/8.3.0 cuda/10.1.168 cmake openmpi/4.0.3-cuda nccl
cd $dir && srun ./benchmark $dim $dim $dim && cd ..
EOF
fi
done
