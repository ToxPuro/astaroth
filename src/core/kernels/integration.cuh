#pragma once

// Choose USE_SMEM or BENCHMARK_RW and generate kernel, then rerun w/ GEN_KERNEL (0)
#define USE_SMEM (0)
#define BENCHMARK_RW (0)
#define GEN_KERNEL (0)

// Device info (TODO GENERIC)
// Use the maximum available reg count per thread
#define REGISTERS_PER_THREAD (255)
#define MAX_REGISTERS_PER_BLOCK (65536)
#if AC_DOUBLE_PRECISION
#define MAX_THREADS_PER_BLOCK (MAX_REGISTERS_PER_BLOCK / REGISTERS_PER_THREAD / 2)
#else
#define MAX_THREADS_PER_BLOCK (MAX_REGISTERS_PER_BLOCK / REGISTERS_PER_THREAD)
#endif

#define UU s, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ
#define AA s, VTXBUF_AX, VTXBUF_AY, VTXBUF_AZ
#define LNRHO s, VTXBUF_LNRHO
#define SS s, VTXBUF_ENTROPY

typedef enum {
    STENCIL_VALUE,
    STENCIL_DERX,
    STENCIL_DERY,
    STENCIL_DERZ,
    STENCIL_DERXX,
    STENCIL_DERYY,
    STENCIL_DERZZ,
    STENCIL_DERXY,
    STENCIL_DERXZ,
    STENCIL_DERYZ,
    NUM_STENCILS,
} Stencils;

#define STENCIL_WIDTH (STENCIL_ORDER + 1)
#define STENCIL_HEIGHT (STENCIL_WIDTH)
#define STENCIL_DEPTH (STENCIL_WIDTH)

#define STENCIL_MIDX (STENCIL_WIDTH / 2 + 1)
#define STENCIL_MIDY (STENCIL_HEIGHT / 2 + 1)
#define STENCIL_MIDZ (STENCIL_DEPTH / 2 + 1)

#define NUM_FIELDS (NUM_VTXBUF_HANDLES)

#define INV_DS (1.0l / 0.04908738521l)

#define DER1_3 (AcReal(INV_DS * 1.0l / 60.0l))
#define DER1_2 (AcReal(INV_DS * -3.0l / 20.0l))
#define DER1_1 (AcReal(INV_DS * 3.0l / 4.0l))
#define DER1_0 (0)

#define DER2_3 (AcReal(INV_DS * INV_DS * 1.0l / 90.0l))
#define DER2_2 (AcReal(INV_DS * INV_DS * -3.0l / 20.0l))
#define DER2_1 (AcReal(INV_DS * INV_DS * 3.0l / 2.0l))
#define DER2_0 (AcReal(INV_DS * INV_DS * -49.0l / 18.0l))

#define DERX_3 (AcReal(INV_DS * INV_DS * 2.0l / 720.0l))
#define DERX_2 (AcReal(INV_DS * INV_DS * -27.0l / 720.0l))
#define DERX_1 (AcReal(INV_DS * INV_DS * 270.0l / 720.0l))
#define DERX_0 (0)

// clang-format off
static __device__ const AcReal
    stencils[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH] =
#include "stencils.in" // Nice hack 8-)

static const AcReal
    host_stencils[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH] =
#include "stencils.in"
;
// clang-format on

typedef struct {
    dim3 tpb;
    int3 dims;
} TBConfig;

static TBConfig getOptimalTBConfig(const int3 dims, VertexBufferArray vba);

static __global__ void
dummy_kernel(void)
{
    DCONST((AcIntParam)0);
    DCONST((AcInt3Param)0);
    DCONST((AcRealParam)0);
    DCONST((AcReal3Param)0);
    acComplex a = exp(AcReal(1) * acComplex(1, 1) * AcReal(1));
    a* a;
}

AcResult
acKernelDummy(void)
{
    dummy_kernel<<<1, 1>>>();
    ERRCHK_CUDA_KERNEL_ALWAYS();
    return AC_SUCCESS;
}

#if USE_SMEM
#if 1
static void
gen_kernel(void)
{
    // kernel unroller
    FILE* fp = fopen("kernel.out", "w");
    ERRCHK_ALWAYS(fp);

    // clang-format off

    fprintf(fp,
        "const int i = blockIdx.x * blockDim.x + start.x - STENCIL_ORDER/2;\n"
        "const int j = blockIdx.y * blockDim.y + start.y - STENCIL_ORDER/2;\n"
        "const int k = blockIdx.z * blockDim.z + start.z - STENCIL_ORDER/2;\n"
        "const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;\n"
        "const int smem_width = blockDim.x + STENCIL_ORDER;\n"
        "const int smem_height = blockDim.y + STENCIL_ORDER;\n");

    for (int field = 0; field < NUM_FIELDS; ++field) {
        for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {

            fprintf(fp,
            "if (tid < smem_width) {\n"
                "\t#pragma unroll\n"
                "\tfor (int height = 0; height < smem_height; ++height)\n"
                "\t\tsmem[tid + height * smem_width] = vba.in[%d][IDX(i + tid, j + height, k + %d)];\n"
            "}\n"
            "__syncthreads();\n", field, depth);


            // WRITE BLOCK START
            fprintf(fp, "if (vertexIdx.x < end.x && vertexIdx.y < end.y && vertexIdx.z < end.z) {\n");
            

            for (int height = 0; height < STENCIL_HEIGHT; ++height) {
                for (int width = 0; width < STENCIL_WIDTH; ++width) {
                    for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
                        if (host_stencils[stencil][depth][height][width] != 0) {       
                            fprintf(fp,
                                    "\tprocessed_stencils[%d][%d] += stencils[%d][%d][%d][%d] * smem[(threadIdx.x + %d) + (threadIdx.y + %d) * smem_width];\n",
                                    field, stencil, stencil, depth, height, width, width, height);
                        }
                    }
                }
            }

            // WRITE BLOCK END
            fprintf(fp, "}\n"); 

            fprintf(fp, "__syncthreads();\n");
        }

        // WRITE BLOCK START
        //fprintf(fp, "if (vertexIdx.x < end.x && vertexIdx.y < end.y && vertexIdx.z < end.z) {\n");

        /*
        const int idx = IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z);
        // vba.out[field][idx] = processed_stencils[field][STENCIL_VALUE];
        // vba.out[field][idx] = processed_stencils[field][STENCIL_DERX];
        // vba.out[field][idx] = processed_stencils[field][STENCIL_DERXX];
        vba.out[field][idx] = processed_stencils[field][STENCIL_DERYZ];
        */
        //fprintf(fp,"vba.out[%d][IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z)] = processed_stencils[%d][STENCIL_DERYZ];\n", field, field);
        //fprintf(fp,"vba.out[%d][IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z)] = processed_stencils[%d][STENCIL_DERYZ];\n", field, field);

        // WRITE BLOCK END
        //fprintf(fp, "}\n");

        #if BENCHMARK_RW
        fprintf(fp, "if (vertexIdx.x < end.x && vertexIdx.y < end.y && vertexIdx.z < end.z) {\n");
        fprintf(fp,
        "\tvba.out[%d][IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z)] = \n"
        "\tprocessed_stencils[%d][STENCIL_VALUE] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERX] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERY] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERZ] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERXX] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERYY] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERZZ] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERXY] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERXZ] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERYZ];\n", field, field, field, field, field, field, field, field, field, field, field);
        fprintf(fp, "}\n");
        #endif

        // clang-format on
    }

    fclose(fp);
}
#else
static void
gen_kernel(void)
{
    // kernel unroller
    FILE* fp = fopen("kernel.out", "w");
    ERRCHK_ALWAYS(fp);

    // clang-format off
    for (int field = 0; field < NUM_FIELDS; ++field) {
        for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {

            fprintf(fp,
            "{const int i = blockIdx.x * blockDim.x + start.x - STENCIL_ORDER/2;\n"
            "const int j = blockIdx.y * blockDim.y + start.y - STENCIL_ORDER/2;\n"
            "const int k = blockIdx.z * blockDim.z + start.z - STENCIL_ORDER/2 + %d;\n"
            "const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;\n"
            "const int smem_width = blockDim.x + STENCIL_ORDER;\n"
            "const int smem_height = blockDim.y + STENCIL_ORDER;\n"
            "if (tid < smem_width) {\n"
                "\tfor (int height = 0; height < smem_height; ++height) {\n"
                "\t\tsmem[tid + height * smem_width] = vba.in[%d][IDX(i + tid, j + height, k)];\n"
                "\t}\n"
            "}\n"
            "__syncthreads();\n", depth, field);


            // WRITE BLOCK START
            fprintf(fp, "if (vertexIdx.x < end.x && vertexIdx.y < end.y && vertexIdx.z < end.z) {\n");
            

            for (int height = 0; height < STENCIL_HEIGHT; ++height) {
                for (int width = 0; width < STENCIL_WIDTH; ++width) {
                    for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
                        if (host_stencils[stencil][depth][height][width] != 0) {       
                            fprintf(fp,
                                    "processed_stencils[%d][%d] += stencils[%d][%d][%d][%d] * smem[(threadIdx.x + %d) + (threadIdx.y + %d) * smem_width];\n",
                                    field, stencil, stencil, depth, height, width, width, height);
                        }
                    }
                }
            }

            // WRITE BLOCK END
            fprintf(fp, "}\n"); 

            fprintf(fp, "__syncthreads();}\n");
        }

        // WRITE BLOCK START
        //fprintf(fp, "if (vertexIdx.x < end.x && vertexIdx.y < end.y && vertexIdx.z < end.z) {\n");

        /*
        const int idx = IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z);
        // vba.out[field][idx] = processed_stencils[field][STENCIL_VALUE];
        // vba.out[field][idx] = processed_stencils[field][STENCIL_DERX];
        // vba.out[field][idx] = processed_stencils[field][STENCIL_DERXX];
        vba.out[field][idx] = processed_stencils[field][STENCIL_DERYZ];
        */
        //fprintf(fp,"vba.out[%d][IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z)] = processed_stencils[%d][STENCIL_DERYZ];\n", field, field);
        //fprintf(fp,"vba.out[%d][IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z)] = processed_stencils[%d][STENCIL_DERYZ];\n", field, field);

        // WRITE BLOCK END
        //fprintf(fp, "}\n");

        #if BENCHMARK_RW
        fprintf(fp, "if (vertexIdx.x < end.x && vertexIdx.y < end.y && vertexIdx.z < end.z) {\n");
        fprintf(fp,
        "\tvba.out[%d][IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z)] = \n"
        "\tprocessed_stencils[%d][STENCIL_VALUE] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERX] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERY] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERZ] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERXX] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERYY] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERZZ] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERXY] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERXZ] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERYZ];\n", field, field, field, field, field, field, field, field, field, field, field);
        fprintf(fp, "}\n");
        #endif

        // clang-format on
    }

    fclose(fp);
}
#endif
#else
static void
gen_kernel(void)
{
    // kernel unroller
    FILE* fp = fopen("kernel.out", "w");
    ERRCHK_ALWAYS(fp);

    // clang-format off
    for (int field = 0; field < NUM_FIELDS; ++field) {
        fprintf(fp, "{\n\tconst AcReal* __restrict__ in = vba.in[%d];\n", field);
        for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
            for (int height = 0; height < STENCIL_HEIGHT; ++height) {
                for (int width = 0; width < STENCIL_WIDTH; ++width) {
                    for (int stencil = 0; stencil < NUM_STENCILS; ++stencil) {
                        if (host_stencils[stencil][depth][height][width] != 0) {
                            fprintf(fp,
                                    "\tprocessed_stencils[%d][%d] += stencils[%d][%d][%d][%d] * in[IDX(vertexIdx.x + (%d), vertexIdx.y + (%d), vertexIdx.z + (%d))];\n",
                                    field, stencil, stencil, depth, height, width,
                                    -STENCIL_ORDER / 2 + width, -STENCIL_ORDER / 2 + height,
                                    -STENCIL_ORDER / 2 + depth);
                        }
                    }
                }
            }
        }
        fprintf(fp, "}\n");

        /*
        const int idx = IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z);
        // vba.out[field][idx] = processed_stencils[field][STENCIL_VALUE];
        // vba.out[field][idx] = processed_stencils[field][STENCIL_DERX];
        // vba.out[field][idx] = processed_stencils[field][STENCIL_DERXX];
        vba.out[field][idx] = processed_stencils[field][STENCIL_DERYZ];
        */
        //fprintf(fp,"vba.out[%d][IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z)] = processed_stencils[%d][STENCIL_DERYZ];\n", field, field);
        //fprintf(fp,"vba.out[%d][IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z)] = processed_stencils[%d][STENCIL_DERYZ];\n", field, field);

        #if BENCHMARK_RW
        fprintf(fp, "if (vertexIdx.x < end.x && vertexIdx.y < end.y && vertexIdx.z < end.z) {\n");
        fprintf(fp,
        "\tvba.out[%d][IDX(vertexIdx.x, vertexIdx.y, vertexIdx.z)] = \n"
        "\tprocessed_stencils[%d][STENCIL_VALUE] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERX] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERY] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERZ] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERXX] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERYY] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERZZ] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERXY] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERXZ] + \n"
        "\tprocessed_stencils[%d][STENCIL_DERYZ];\n", field, field, field, field, field, field, field, field, field, field, field);
        fprintf(fp, "}\n");
        #endif

        // clang-format on
    }

    fclose(fp);
}
#endif // USE_SMEM

static __device__ __forceinline__ AcReal
value(const AcReal s[NUM_FIELDS][NUM_STENCILS], const VertexBufferHandle handle)
{
    return s[handle][STENCIL_VALUE];
}

static __device__ __forceinline__ AcReal3
value(const AcReal s[NUM_FIELDS][NUM_STENCILS], const VertexBufferHandle x,
      const VertexBufferHandle y, const VertexBufferHandle z)
{
    return (AcReal3){
        s[x][STENCIL_VALUE],
        s[y][STENCIL_VALUE],
        s[z][STENCIL_VALUE],
    };
}

static __device__ __forceinline__ AcReal3
gradient(const AcReal s[NUM_FIELDS][NUM_STENCILS], const VertexBufferHandle handle)
{
    return (AcReal3){
        s[handle][STENCIL_DERX],
        s[handle][STENCIL_DERY],
        s[handle][STENCIL_DERZ],
    };
}

static __device__ __forceinline__ AcMatrix
gradients(const AcReal s[NUM_FIELDS][NUM_STENCILS], const VertexBufferHandle x,
          const VertexBufferHandle y, const VertexBufferHandle z)
{
    return (AcMatrix){gradient(s, x), gradient(s, y), gradient(s, z)};
}

static __device__ __forceinline__ AcReal
divergence(const AcReal s[NUM_FIELDS][NUM_STENCILS], const VertexBufferHandle x,
           const VertexBufferHandle y, const VertexBufferHandle z)
{
    return s[x][STENCIL_DERX] + s[y][STENCIL_DERY] + s[z][STENCIL_DERZ];
}

static __device__ __forceinline__ AcReal3
curl(const AcReal s[NUM_FIELDS][NUM_STENCILS], const VertexBufferHandle x,
     const VertexBufferHandle y, const VertexBufferHandle z)
{
    return (AcReal3){
        s[z][STENCIL_DERY] - s[y][STENCIL_DERZ],
        s[x][STENCIL_DERZ] - s[z][STENCIL_DERX],
        s[y][STENCIL_DERX] - s[x][STENCIL_DERY],
    };
}

static __device__ __forceinline__ AcReal
laplace(const AcReal s[NUM_FIELDS][NUM_STENCILS], const VertexBufferHandle handle)
{
    return s[handle][STENCIL_DERXX] + s[handle][STENCIL_DERYY] + s[handle][STENCIL_DERZZ];
}

static __device__ __forceinline__ AcReal3
laplace(const AcReal s[NUM_FIELDS][NUM_STENCILS], const VertexBufferHandle x,
        const VertexBufferHandle y, const VertexBufferHandle z)
{
    return (AcReal3){laplace(s, x), laplace(s, y), laplace(s, z)};
}

static __device__ __forceinline__ AcReal3
induction(const AcReal s[NUM_FIELDS][NUM_STENCILS])
{
    const AcReal3 B   = curl(AA);
    const AcReal3 lap = laplace(AA);
    const AcReal3 uu  = value(UU);

    return cross(uu, B) + DCONST(AC_eta) * lap;
}

static __device__ __forceinline__ AcMatrix
stress_tensor(const AcReal s[NUM_FIELDS][NUM_STENCILS], const VertexBufferHandle x,
              const VertexBufferHandle y, const VertexBufferHandle z)
{
    AcMatrix S;

    S.row[0].x = (AcReal)(2. / 3.) * s[x][STENCIL_DERX] -
                 (AcReal)(1. / 3.) * (s[y][STENCIL_DERY] + s[z][STENCIL_DERZ]);
    S.row[0].y = (AcReal)(1. / 2.) * (s[x][STENCIL_DERY] + s[y][STENCIL_DERX]);
    S.row[0].z = (AcReal)(1. / 2.) * (s[x][STENCIL_DERZ] + s[z][STENCIL_DERX]);

    S.row[1].y = (AcReal)(2. / 3.) * s[y][STENCIL_DERY] -
                 (AcReal)(1. / 3.) * (s[x][STENCIL_DERX] + s[z][STENCIL_DERZ]);

    S.row[1].z = (AcReal)(1. / 2.) * (s[y][STENCIL_DERZ] + s[z][STENCIL_DERY]);

    S.row[2].z = (AcReal)(2. / 3.) * s[z][STENCIL_DERZ] -
                 (AcReal)(1. / 3.) * (s[x][STENCIL_DERX] + s[y][STENCIL_DERY]);

    S.row[1].x = S.row[0].y;
    S.row[2].x = S.row[0].z;
    S.row[2].y = S.row[1].z;

    return S;
}

static __device__ __forceinline__ AcReal3
gradient_of_divergence(const AcReal s[NUM_FIELDS][NUM_STENCILS], const VertexBufferHandle x,
                       const VertexBufferHandle y, const VertexBufferHandle z)
{
    return (AcReal3){
        s[x][STENCIL_DERXX] + s[y][STENCIL_DERXY] + s[z][STENCIL_DERXZ],
        s[x][STENCIL_DERXY] + s[y][STENCIL_DERYY] + s[z][STENCIL_DERYZ],
        s[x][STENCIL_DERXZ] + s[y][STENCIL_DERYZ] + s[z][STENCIL_DERZZ],
    };
}

static __device__ __forceinline__ AcReal
contract(const AcMatrix mat)
{
    return dot(mat.row[0], mat.row[0]) + dot(mat.row[1], mat.row[1]) + dot(mat.row[2], mat.row[2]);
}

static __device__ AcReal
continuity(const AcReal s[NUM_FIELDS][NUM_STENCILS])
{
    const AcReal3 uu         = value(UU);
    const AcReal3 grad_lnrho = gradient(LNRHO);
    const AcReal div_uu      = divergence(UU);

    return -dot(uu, grad_lnrho) - div_uu;
}

static __device__ AcReal3
momentum(const AcReal s[NUM_FIELDS][NUM_STENCILS])
{
    const AcReal3 uu       = value(UU);
    const AcMatrix grad_uu = gradients(UU);
    const AcMatrix S       = stress_tensor(UU);
    // NOTE WARNING AC_cs2_sound NOT PROPERLY LOADED TODO FIX
    const AcReal cs2_sound = DCONST(AC_cs_sound) * DCONST(AC_cs_sound);
    const AcReal cs2       = cs2_sound * exp(DCONST(AC_gamma) * value(SS) / DCONST(AC_cp_sound) +
                                       (DCONST(AC_gamma) - 1) * (value(LNRHO) - DCONST(AC_lnrho0)));

    const AcReal3 j = ((AcReal)(1.) / DCONST(AC_mu0)) * (gradient_of_divergence(AA) - laplace(AA));
    const AcReal3 B = curl(AA);
    const AcReal inv_rho = (AcReal)(1.) / exp(value(LNRHO));

    const AcReal3 mom = -mul(grad_uu, uu) -
                        cs2 *
                            ((AcReal(1.0) / DCONST(AC_cp_sound)) * gradient(SS) + gradient(LNRHO)) +
                        inv_rho * cross(j, B) +
                        DCONST(AC_nu_visc) *
                            (laplace(UU) +
                             (AcReal(1.0) / AcReal(3.0)) * gradient_of_divergence(UU) +
                             AcReal(2.0) * mul(S, gradient(LNRHO))) +
                        DCONST(AC_zeta) * gradient_of_divergence(UU);

    return mom;
}

static __device__ __forceinline__ AcReal
lnT(const AcReal s[NUM_FIELDS][NUM_STENCILS])
{
    return DCONST(AC_lnT0) + DCONST(AC_gamma) * value(SS) / DCONST(AC_cp_sound) +
           (DCONST(AC_gamma) - AcReal(1.0)) * (value(LNRHO) - DCONST(AC_lnrho0));
}

static __device__ AcReal
heat_conduction(const AcReal s[NUM_FIELDS][NUM_STENCILS])
{
    const AcReal inv_AC_cp_sound = AcReal(1.0) / DCONST(AC_cp_sound);
    const AcReal3 grad_ln_chi    = -gradient(LNRHO);
    const AcReal first_term      = DCONST(AC_gamma) * inv_AC_cp_sound * laplace(SS) +
                              (DCONST(AC_gamma) - AcReal(1.0)) * laplace(LNRHO);
    const AcReal3 second_term = DCONST(AC_gamma) * inv_AC_cp_sound * gradient(SS) +
                                (DCONST(AC_gamma) - AcReal(1.0)) * gradient(LNRHO);
    const AcReal3 third_term = DCONST(AC_gamma) *
                                   (inv_AC_cp_sound * gradient(SS) + gradient(LNRHO)) +
                               grad_ln_chi;
    const AcReal chi = (AcReal(0.001)) / (exp(value(LNRHO)) * DCONST(AC_cp_sound));
    return DCONST(AC_cp_sound) * chi * (first_term + dot(second_term, third_term));
}

static __device__ AcReal
entropy(const AcReal s[NUM_FIELDS][NUM_STENCILS])
{
    const AcMatrix S    = stress_tensor(UU);
    const AcReal inv_pT = AcReal(1.0) / (exp(value(LNRHO)) * exp(lnT(s)));
    const AcReal3 j  = (AcReal(1.0) / DCONST(AC_mu0)) * (gradient_of_divergence(AA) - laplace(AA));
    const AcReal RHS = (0) - (0) + DCONST(AC_eta) * (DCONST(AC_mu0)) * dot(j, j) +
                       AcReal(2.0) * exp(value(LNRHO)) * DCONST(AC_nu_visc) * contract(S) +
                       DCONST(AC_zeta) * exp(value(LNRHO)) * divergence(UU) * divergence(UU);
    return -dot(value(UU), gradient(SS)) + inv_pT * RHS + heat_conduction(s);
}

template <int step_number>
static __device__ __forceinline__ AcReal
rk3_integrate(const AcReal state_previous, const AcReal state_current, const AcReal rate_of_change,
              const AcReal dt)
{
    // Williamson (1980)
    const AcReal alpha[] = {0, AcReal(.0), AcReal(-5. / 9.), AcReal(-153. / 128.)};
    const AcReal beta[]  = {0, AcReal(1. / 3.), AcReal(15. / 16.), AcReal(8. / 15.)};

    // Note the indexing: +1 to avoid an unnecessary warning about "out-of-bounds"
    // access (when accessing beta[step_number-1] even when step_number >= 1)
    switch (step_number) {
    case 0:
        return state_current + beta[step_number + 1] * rate_of_change * dt;
    case 1: // Fallthrough
    case 2:
        return state_current +
               beta[step_number + 1] * (alpha[step_number + 1] * (AcReal(1.) / beta[step_number]) *
                                            (state_current - state_previous) +
                                        rate_of_change * dt);
    default:
        return NAN;
    }
}

template <int step_number>
static __device__ void
calc_roc(const AcReal s[NUM_FIELDS][NUM_STENCILS], AcReal rate_of_change[NUM_FIELDS],
         VertexBufferArray vba, const int idx)
{
    int handle;

    const AcReal3 uu   = value(UU);
    const AcReal3 aa   = value(AA);
    const AcReal lnrho = value(LNRHO);
    const AcReal ss    = value(SS);

    const AcReal3 grad_lnrho = gradient(LNRHO);
    const AcReal div_uu      = divergence(UU);

    const AcReal3 B          = curl(AA);
    const AcReal3 laplace_aa = laplace(AA);

    const AcReal cs_sound  = DCONST(AC_cs_sound);
    const AcReal gamma     = DCONST(AC_gamma);
    const AcReal cp_sound  = DCONST(AC_cp_sound);
    const AcReal lnrho0    = DCONST(AC_lnrho0);
    const AcReal mu0       = DCONST(AC_mu0);
    const AcReal cs2_sound = cs_sound * cs_sound;

    const AcMatrix grad_uu = gradients(UU);
    const AcMatrix S       = stress_tensor(UU);

    const AcReal cs2 = cs2_sound * exp(gamma * ss / cp_sound + (gamma - 1) * (lnrho - lnrho0));

    const AcReal3 grad_div_aa = gradient_of_divergence(AA);
    const AcReal3 j           = (grad_div_aa - laplace_aa) / mu0;

    const AcReal nu_visc      = DCONST(AC_nu_visc);
    const AcReal3 laplace_uu  = laplace(UU);
    const AcReal3 grad_div_uu = gradient_of_divergence(UU);
    const AcReal zeta         = DCONST(AC_zeta);
    const AcReal3 grad_ss     = gradient(SS);

    const AcReal lnT0          = DCONST(AC_lnT0);
    const AcReal laplace_lnrho = laplace(LNRHO);
    const AcReal laplace_ss    = laplace(SS);

    const AcReal3 grad_ln_chi = -grad_lnrho;
    const AcReal first_term   = gamma * laplace_ss / cp_sound + (gamma - 1) * laplace_lnrho;
    const AcReal3 second_term = gamma * grad_ss / cp_sound + (gamma - 1) * grad_lnrho;
    const AcReal3 third_term  = gamma * (grad_ss / cp_sound + grad_lnrho) + grad_ln_chi;
    const AcReal chi          = (AcReal(0.001)) / (exp(lnrho) * cp_sound);

    rate_of_change[VTXBUF_LNRHO] = -dot(uu, grad_lnrho) - div_uu;
    handle                       = VTXBUF_LNRHO;
    vba.out[handle][idx]         = rk3_integrate<step_number>(vba.out[handle][idx],
                                                      s[handle][STENCIL_VALUE],
                                                      rate_of_change[handle], DCONST(AC_dt));

    const AcReal3 ind         = cross(uu, B) + DCONST(AC_eta) * laplace_aa;
    rate_of_change[VTXBUF_AX] = ind.x;
    rate_of_change[VTXBUF_AY] = ind.y;
    rate_of_change[VTXBUF_AZ] = ind.z;
    handle                    = VTXBUF_AX;
    vba.out[handle][idx]      = rk3_integrate<step_number>(vba.out[handle][idx],
                                                      s[handle][STENCIL_VALUE],
                                                      rate_of_change[handle], DCONST(AC_dt));
    handle                    = VTXBUF_AY;
    vba.out[handle][idx]      = rk3_integrate<step_number>(vba.out[handle][idx],
                                                      s[handle][STENCIL_VALUE],
                                                      rate_of_change[handle], DCONST(AC_dt));
    handle                    = VTXBUF_AZ;
    vba.out[handle][idx]      = rk3_integrate<step_number>(vba.out[handle][idx],
                                                      s[handle][STENCIL_VALUE],
                                                      rate_of_change[handle], DCONST(AC_dt));

    const AcReal3 mom = -mul(grad_uu, uu) - cs2 * (grad_ss / cp_sound + grad_lnrho) +
                        cross(j, B) / exp(lnrho) +
                        nu_visc * (laplace_uu + grad_div_uu / AcReal(3.0) +
                                   AcReal(2.0) * mul(S, grad_lnrho)) +
                        zeta * grad_div_uu;

    rate_of_change[VTXBUF_UUX] = mom.x;
    rate_of_change[VTXBUF_UUY] = mom.y;
    rate_of_change[VTXBUF_UUZ] = mom.z;
    handle                     = VTXBUF_UUX;
    vba.out[handle][idx]       = rk3_integrate<step_number>(vba.out[handle][idx],
                                                      s[handle][STENCIL_VALUE],
                                                      rate_of_change[handle], DCONST(AC_dt));
    handle                     = VTXBUF_UUY;
    vba.out[handle][idx]       = rk3_integrate<step_number>(vba.out[handle][idx],
                                                      s[handle][STENCIL_VALUE],
                                                      rate_of_change[handle], DCONST(AC_dt));
    handle                     = VTXBUF_UUZ;
    vba.out[handle][idx]       = rk3_integrate<step_number>(vba.out[handle][idx],
                                                      s[handle][STENCIL_VALUE],
                                                      rate_of_change[handle], DCONST(AC_dt));

    const AcReal lnT             = lnT0 + gamma * ss / cp_sound + (gamma - 1) * (lnrho - lnrho0);
    const AcReal heat_conduction = cp_sound * chi * (first_term + dot(second_term, third_term));

    const AcReal eta           = DCONST(AC_eta);
    const AcReal divergence_uu = divergence(UU);
    const AcReal inv_pT        = 1 / (exp(lnrho) * exp(lnT));
    const AcReal RHS = (0) - (0) + eta * mu0 * dot(j, j) + 2 * exp(lnrho) * nu_visc * contract(S) +
                       zeta * exp(lnrho) * divergence_uu * divergence_uu;

    rate_of_change[VTXBUF_ENTROPY] = -dot(uu, grad_ss) + inv_pT * RHS + heat_conduction;
    // const AcReal entr              = entropy(s);
    // rate_of_change[VTXBUF_ENTROPY] = entr;
    handle               = VTXBUF_ENTROPY;
    vba.out[handle][idx] = rk3_integrate<step_number>(vba.out[handle][idx],
                                                      s[handle][STENCIL_VALUE],
                                                      rate_of_change[handle], DCONST(AC_dt));
}

template <int step_number>
static __global__ __launch_bounds__(MAX_THREADS_PER_BLOCK) //
    void solve(const int3 start, const int3 end, VertexBufferArray vba)
{
#if USE_SMEM
    extern __shared__ AcReal smem[];
#endif // USE_SMEM

    const int3 vertexIdx = (int3){
        threadIdx.x + blockIdx.x * blockDim.x + start.x,
        threadIdx.y + blockIdx.y * blockDim.y + start.y,
        threadIdx.z + blockIdx.z * blockDim.z + start.z,
    };
    const int3 globalVertexIdx = (int3){
        d_multigpu_offset.x + vertexIdx.x,
        d_multigpu_offset.y + vertexIdx.y,
        d_multigpu_offset.z + vertexIdx.z,
    };

#if USE_SMEM // Need all threads to participate and not return early
    assert(blockDim.x * blockDim.y * blockDim.z >= blockDim.x + STENCIL_ORDER);
#else
    if (vertexIdx.x >= end.x || vertexIdx.y >= end.y || vertexIdx.z >= end.z)
        return;

    assert(vertexIdx.x < DCONST(AC_nx_max) && vertexIdx.y < DCONST(AC_ny_max) &&
           vertexIdx.z < DCONST(AC_nz_max));

    assert(vertexIdx.x >= DCONST(AC_nx_min) && vertexIdx.y >= DCONST(AC_ny_min) &&
           vertexIdx.z >= DCONST(AC_nz_min));
#endif

    AcReal processed_stencils[NUM_FIELDS][NUM_STENCILS] = {0};
#if !GEN_KERNEL
#include "kernel.out"
#endif

    const int idx = IDX(vertexIdx);

#if !BENCHMARK_RW
#if USE_SMEM // WRITE BLOCK START
    if (vertexIdx.x < end.x && vertexIdx.y < end.y && vertexIdx.z < end.z) {
#endif
        AcReal rate_of_change[NUM_FIELDS] = {0};
        // calc_roc<step_number>(processed_stencils, rate_of_change, vba, idx);

        const AcReal cont = continuity(processed_stencils);
        const AcReal3 mom = momentum(processed_stencils);
        const AcReal3 ind = induction(processed_stencils);
        const AcReal entr = entropy(processed_stencils);

        rate_of_change[VTXBUF_LNRHO]   = cont;
        rate_of_change[VTXBUF_UUX]     = mom.x;
        rate_of_change[VTXBUF_UUY]     = mom.y;
        rate_of_change[VTXBUF_UUZ]     = mom.z;
        rate_of_change[VTXBUF_AX]      = ind.x;
        rate_of_change[VTXBUF_AY]      = ind.y;
        rate_of_change[VTXBUF_AZ]      = ind.z;
        rate_of_change[VTXBUF_ENTROPY] = entr;

        // Manually unrolled
        AcReal* __restrict__ out0 = vba.out[0];
        AcReal* __restrict__ out1 = vba.out[1];
        AcReal* __restrict__ out2 = vba.out[2];
        AcReal* __restrict__ out3 = vba.out[3];
        AcReal* __restrict__ out4 = vba.out[4];
        AcReal* __restrict__ out5 = vba.out[5];
        AcReal* __restrict__ out6 = vba.out[6];
        AcReal* __restrict__ out7 = vba.out[7];
        out0[idx] = rk3_integrate<step_number>(out0[idx], processed_stencils[0][STENCIL_VALUE],
                                               rate_of_change[0], DCONST(AC_dt));
        out1[idx] = rk3_integrate<step_number>(out1[idx], processed_stencils[1][STENCIL_VALUE],
                                               rate_of_change[1], DCONST(AC_dt));
        out2[idx] = rk3_integrate<step_number>(out2[idx], processed_stencils[2][STENCIL_VALUE],
                                               rate_of_change[2], DCONST(AC_dt));
        out3[idx] = rk3_integrate<step_number>(out3[idx], processed_stencils[3][STENCIL_VALUE],
                                               rate_of_change[3], DCONST(AC_dt));
        out4[idx] = rk3_integrate<step_number>(out4[idx], processed_stencils[4][STENCIL_VALUE],
                                               rate_of_change[4], DCONST(AC_dt));
        out5[idx] = rk3_integrate<step_number>(out5[idx], processed_stencils[5][STENCIL_VALUE],
                                               rate_of_change[5], DCONST(AC_dt));
        out6[idx] = rk3_integrate<step_number>(out6[idx], processed_stencils[6][STENCIL_VALUE],
                                               rate_of_change[6], DCONST(AC_dt));
        out7[idx] = rk3_integrate<step_number>(out7[idx], processed_stencils[7][STENCIL_VALUE],
                                               rate_of_change[7], DCONST(AC_dt));

        /*
    #pragma unroll
            for (int i = 0; i < NUM_FIELDS; ++i)
                vba.out[i][idx] = rk3_integrate<step_number>(vba.out[i][idx],
                                                             processed_stencils[i][STENCIL_VALUE],
                                                             rate_of_change[i], DCONST(AC_dt));
        */
#if USE_SMEM // WRITE BLOCK END
    }
#endif
#endif // BENCHMARK_RW
}

AcResult
acKernelIntegrateSubstep(const cudaStream_t stream, const int step_number, const int3 start,
                         const int3 end, VertexBufferArray vba)
{
    // optimal integration dims in a hash table (hashes start & end indices) + timeout
    // Mesh dimensions available with DCONST(AC_nx) etc.

    // For field f
    //  For stencil depth k
    //      For stencil height j
    //          For stencil width i
    //              For stencils w
    //                  processed_stencils[w] = stencils[w][k][j][i] *
    //            field[f][vertexIdx.z - MIDZ + k][vertexIdx.y - MIDY + j][vertexIdx.x - MIDX +
    //            i];
    //
    // Even better, template magic unroll and skip if stencils[w][k][j][i] == 0
    //
    // Test also space filling! if time
    //
    // ALSO NEW MENTALITY! We do not have preprocessed any more! We have actual
    // physical stencils! User sets stencils and does stencil fetches!
    //
    // Stencil value = {
    //      [-1][0] = 60.0 / 30.0,
    //      [0][0] = 1.0,
    // };
    // Field a attach value, derx; // VALUE IS ALWAYS IMPLICITLY ATTACHED
    //
    // value(a) * something
    // derx(a) * something
    //
    // ALSO CONSISTENCY! a is either a value or handle, not both
    //
    // OTA TEMPLATE STEP NUMBER MYOS POIS! RK3 from DSL stdlib!

    ERRCHK_ALWAYS(step_number >= 0);
    ERRCHK_ALWAYS(step_number < 3);
    const dim3 tpb = getOptimalTBConfig(end - start, vba).tpb;
#if USE_SMEM
    const size_t smem = (tpb.x + STENCIL_ORDER) * (tpb.y + STENCIL_ORDER) * sizeof(AcReal);
#else
    const size_t smem = 0;
#endif

    const int3 n = end - start;
    const dim3 bpg((unsigned int)ceil(n.x / AcReal(tpb.x)), //
                   (unsigned int)ceil(n.y / AcReal(tpb.y)), //
                   (unsigned int)ceil(n.z / AcReal(tpb.z)));

    if (step_number == 0)
        solve<0><<<bpg, tpb, smem, stream>>>(start, end, vba);
    else if (step_number == 1)
        solve<1><<<bpg, tpb, smem, stream>>>(start, end, vba);
    else
        solve<2><<<bpg, tpb, smem, stream>>>(start, end, vba);

    ERRCHK_CUDA_KERNEL();

    return AC_SUCCESS;
}

AcResult
acKernelAutoOptimizeIntegration(const int3 start, const int3 end, VertexBufferArray vba)
{

// DEBUG UNROLL
#if GEN_KERNEL
    gen_kernel(); // Debug TODO REMOVE
    exit(0);
#endif
    //// DEBUG UNROLL
    (void)start;
    (void)end;
    (void)vba;
    fprintf(stderr, "acKernelAutoOptimizeIntegration is deprecated\n");
    return AC_FAILURE;
}

static TBConfig
autotune(const int3 dims, VertexBufferArray vba)
{
    fprintf(stderr, "------------------TODO WARNING FIX autotune HARMFUL!----------------\ndt not "
                    "set properly and MUST call w/ all possible subdomain sizes before actual "
                    "simulation with dt = 0, otherwise advances the simulation arbitrarily!!\n");

    const int3 start = (int3){NGHOST, NGHOST, NGHOST};
    const int3 end   = start + dims;

    printf("Autotuning for (%d, %d, %d)... ", dims.x, dims.y, dims.z);
    // RK3
    dim3 best_dims(0, 0, 0);
    float best_time          = INFINITY;
    const int num_iterations = 10;

#if USE_SMEM
    for (int z = 1; z <= 1; ++z) { // TODO CHECK Z GOES ONLY TO 1
#else
    for (int z = 1; z <= MAX_THREADS_PER_BLOCK; ++z) {
#endif
        for (int y = 1; y <= MAX_THREADS_PER_BLOCK; ++y) {
            for (int x = 4; x <= MAX_THREADS_PER_BLOCK; x += 4) {

                // if (x > end.x - start.x || y > end.y - start.y || z > end.z - start.z)
                //    break;

                if (x * y * z > MAX_THREADS_PER_BLOCK)
                    break;

                if (x * y * z * REGISTERS_PER_THREAD > MAX_REGISTERS_PER_BLOCK)
                    break;

                    // if (((x * y * z) % WARP_SIZE) != 0)
                    //    continue;

#if USE_SMEM
                if ((x * y * z) < x + STENCIL_ORDER) // WARNING NOTE: Only use if using smem
                    continue;
#endif

                const dim3 tpb(x, y, z);
                const int3 n = end - start;
                const dim3 bpg((unsigned int)ceil(n.x / AcReal(tpb.x)), //
                               (unsigned int)ceil(n.y / AcReal(tpb.y)), //
                               (unsigned int)ceil(n.z / AcReal(tpb.z)));
#if USE_SMEM
                const size_t smem = (tpb.x + STENCIL_ORDER) * (tpb.y + STENCIL_ORDER) *
                                    sizeof(AcReal);
#else
                const size_t smem = 0;
#endif

                cudaDeviceSynchronize();
                if (cudaGetLastError() != cudaSuccess) // resets the error if any
                    continue;

                // printf("(%d, %d, %d)\n", x, y, z);

                cudaEvent_t tstart, tstop;
                cudaEventCreate(&tstart);
                cudaEventCreate(&tstop);

                cudaEventRecord(tstart); // ---------------------------------------- Timing start
                for (int i = 0; i < num_iterations; ++i)
                    solve<2><<<bpg, tpb, smem, 0>>>(start, end, vba);

                cudaEventRecord(tstop); // ----------------------------------------- Timing end
                cudaEventSynchronize(tstop);
                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, tstart, tstop);

                ERRCHK_CUDA_KERNEL_ALWAYS();
                // printf("(%d, %d, %d): %.4g ms\n", x, y, z, (double)milliseconds /
                // num_iterations); fflush(stdout);
                if (milliseconds < best_time) {
                    best_time = milliseconds;
                    best_dims = tpb;
                }
            }
        }
    }
    printf("\x1B[32m%s\x1B[0m\n", "OK!");
    fflush(stdout);
    //#if AC_VERBOSE
    printf("Auto-optimization done. The best threadblock dimensions for rkStep (%d, %d, %d): "
           "(%d, "
           "%d, %d) "
           "%f "
           "ms\n",
           dims.x, dims.y, dims.z, best_dims.x, best_dims.y, best_dims.z,
           double(best_time) / num_iterations);
    //#endif

    // Failed to find valid thread block dimensions
    ERRCHK_ALWAYS(best_dims.x * best_dims.y * best_dims.z > 0);

    TBConfig c;
    c.tpb  = best_dims;
    c.dims = dims;
    return c;
}

#include <vector>
static std::vector<TBConfig> tbconfigs;

static TBConfig
getOptimalTBConfig(const int3 dims, VertexBufferArray vba)
{
    for (auto c : tbconfigs) {
        if (c.dims == dims)
            return c;
    }
    TBConfig c = autotune(dims, vba);
    tbconfigs.push_back(c);
    return c;
}