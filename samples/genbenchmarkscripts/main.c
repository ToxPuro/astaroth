#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int
main(void)
{
    const int max_nprocs = 128;
    for (int nprocs = 1; nprocs <= max_nprocs; nprocs *= 2) {
        char filename[4096];
        sprintf(filename, "benchmark_%d.sh", nprocs);

        FILE* fp = fopen(filename, "w");
        assert(fp);

        // Boilerplate
        fprintf(fp, "#!/bin/bash\n");
        fprintf(fp, "#BATCH --job-name=astaroth\n");
        fprintf(fp, "#SBATCH --account=project_2000403\n");
        fprintf(fp, "#SBATCH --time=00:14:59\n");
        fprintf(fp, "#SBATCH --mem=32000\n");
        fprintf(fp, "#SBATCH --partition=gpu\n");

        // nprocs, nodes, gpus
        const int max_gpus_per_node = 4;
        const int gpus_per_node     = nprocs < max_gpus_per_node ? nprocs : max_gpus_per_node;
        const int nodes             = (int)ceil((double)nprocs / max_gpus_per_node);
        fprintf(fp, "#SBATCH --gres=gpu:v100:%d\n", gpus_per_node);
        fprintf(fp, "#SBATCH -n %d\n", nprocs);
        fprintf(fp, "#SBATCH -N %d\n", nodes);

        // Modules
        fprintf(fp, "module load gcc/8.3.0 cuda/10.1.168 cmake hpcx-mpi/2.5.0-cuda nccl\n");
        fprintf(fp, "export UCX_MEMTYPE_CACHE=n\n");

        // Profile and run
        fprintf(fp, "mkdir -p profile_%d\n", nprocs);
        fprintf(fp, "srun nvprof --annotate-mpi openmpi -o profile_%d/%%p.nvprof ./benchmark\n",
                nprocs);

        fclose(fp);
    }

    return EXIT_SUCCESS;
}
