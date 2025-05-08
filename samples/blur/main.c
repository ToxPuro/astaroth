#include <float.h> // FLT_EPSILON
#include <stdlib.h>

#include "astaroth.h"
#include "astaroth_utils.h"

#include "timer_hires.h"

/**
    Building:
        cmake -DPROGRAM_MODULE_DIR=<path/to/this/file> -DDSL_MODULE_DIR=<path/to/blur.ac>
   -DBUILD_MHD_SAMPLES=OFF -DBUILD_STANDALONE=OFF ..

   F.ex.
   cmake -DBUILD_STANDALONE=OFF -DBUILD_MHD_SAMPLES=OFF -DPROGRAM_MODULE_DIR=../samples/blur
   -DDSL_MODULE_DIR=../acc-runtime/samples/blur ..
*/

#define NSAMPLES (100)

int
main(int argc, char** argv)
{
    AcMeshInfo info;

    // Set mesh dimensions
    if (argc != 3) {
        fprintf(stderr, "Usage: ./blur <nx> <ny>\n");
        return EXIT_FAILURE;
    }
    else {
        info.int3_params[AC_nlocal] = (int3){atoi(argv[1]),atoi(argv[2]),1};
        acHostUpdateParams(&info);
    }

    AcMesh mesh;
    acHostMeshCreate(info, &mesh);
    // acHostMeshRandomize(&mesh);
    acHostMeshSet(1, &mesh);

    Device device;
    acDeviceCreate(0, info, &device);
    acDevicePrintInfo(device);
    acDeviceLoadMesh(device, STREAM_DEFAULT, mesh);

    // Benchmark
    Timer t;
    timer_reset(&t);

    const int3 nn_min = (int3){info.int3_params[AC_nmin].x, info.int3_params[AC_nmin].y,
                               info.int3_params[AC_nmin].z};
    const int3 nn_max = (int3){info.int3_params[AC_nlocal_max].x, info.int3_params[AC_nlocal_max].y,
                               info.int3_params[AC_nlocal_max].z};
    const int3 start  = nn_min;
    const int3 end    = nn_max;

    acDeviceLaunchKernel(device, STREAM_DEFAULT, blur_kernel, 
		    (Volume){(size_t)start.x,(size_t)start.y,(size_t)start.z}, 
		    (Volume){(size_t)end.x,(size_t)end.y,(size_t)end.z}
		    );
    acDeviceSwapBuffers(device);

    acDeviceStoreMesh(device, STREAM_DEFAULT, &mesh);
    printf("Store complete\n");

    const bool print_bounds = true;
    for (size_t w = 0; w < NUM_VTXBUF_HANDLES; ++w) {
        for (int k = nn_min.z; k < nn_max.z; ++k) {
            printf("==== DEPTH %d ====\n", k);
            if (print_bounds) {
                for (int j = 0; j < info.int3_params[AC_mlocal].y; ++j) {
                    for (int i = 0; i < info.int3_params[AC_mlocal].x; ++i) {
                        printf("%.3g ",
                               (double)mesh.vertex_buffer[w][acVertexBufferIdx(i, j, k, info)]);
                    }
                    printf("\n");
                }
            }
            else {
                for (int j = nn_min.y; j < nn_max.y; ++j) {
                    for (int i = nn_min.x; i < nn_max.x; ++i) {
                        printf("%.3g ",
                               (double)mesh.vertex_buffer[w][acVertexBufferIdx(i, j, k, info)]);
                    }
                    printf("\n");
                }
            }
        }
    }

    acDeviceDestroy(&device);
    acHostMeshDestroy(&mesh);

    return EXIT_SUCCESS;
}
