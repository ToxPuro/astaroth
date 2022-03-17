#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include <math.h>

#include <errno.h>
#include <string.h>
#include <time.h>

#define ERROR(str)                                                                                 \
    {                                                                                              \
        time_t terr;                                                                               \
        time(&terr);                                                                               \
        fprintf(stderr, "%s", ctime(&terr));                                                       \
        fprintf(stderr, "\tError in file %s line %d: %s\n", __FILE__, __LINE__, str);              \
        fprintf(stderr, "\tErrno: `%s`\n", errno ? strerror(errno) : "");                          \
        fflush(stderr);                                                                            \
        exit(EXIT_FAILURE);                                                                        \
        abort();                                                                                   \
    }

#define ERRCHK(retval)                                                                             \
    {                                                                                              \
        if (!(retval))                                                                             \
            ERROR(#retval " was false");                                                           \
    }

typedef double real;

typedef struct {
    size_t x, y, z;
} Dim;

void
printDim(const char* id, const Dim dim)
{
    printf("%s: (%lu, %lu, %lu)\n", id, dim.x, dim.y, dim.z);
}

typedef struct {
    Dim mm;
    real* data;
} Block;

void
printNelems(const real* data, const Dim mm, const Dim offset, const size_t nelems)
{
    ERRCHK(offset.x + nelems < mm.x);
    for (size_t i = 0; i < nelems; ++i) {
        printf("%g, ", data[i + offset.x + offset.y * mm.x + offset.z * mm.x * mm.y]);
    }
}

bool
is_valid(const real* data, const size_t n, const real expected_range)
{
    ERRCHK(expected_range > 0);
    for (size_t i = 0; i < n; ++i)
        if (!(fabs(data[i]) < expected_range))
            return false;

    return true;
}

void
printBlock(const char* id, const Block block)
{
    printf("Block `%s`\n", id);
    printDim("\tmm", block.mm);
    printf("\tdata: %p\n", block.data);

    const size_t nelems = 5;
    printf("\tData start: ");
    printNelems(block.data, block.mm, (Dim){0, 0, 0}, nelems);
    printf("\n");

    printf("\tData mid: ");
    printNelems(block.data, block.mm, (Dim){block.mm.x / 2, block.mm.y / 2, block.mm.z / 2},
                nelems);
    printf("\n");

    printf("\tData End: ");
    printNelems(block.data, block.mm,
                (Dim){block.mm.x - nelems - 1, block.mm.y - nelems - 1, block.mm.z - nelems - 1},
                nelems);
    printf("\n");

    printf("\tData is valid? %d\n",
           is_valid(block.data, block.mm.x * block.mm.y * block.mm.z, 1e10));
}

Block
blockCreate(const Dim mm)
{
    Block block = {
        .mm   = mm,
        .data = NULL,
    };
    block.data = malloc(sizeof(block.data[0]) * mm.x * mm.y * mm.z);
    ERRCHK(block.data);

    return block;
}

void
blockDestroy(Block* block)
{
    block->mm = (Dim){0, 0, 0};
    free(block->data);
    block->data = NULL;
}

/*
static bool
dim_less_than(const Dim a, const Dim b)
{
    return (a.x < b.x) && (a.y < b.y) && (a.z < b.z);
}

static Dim
dim_add(const Dim a, const Dim b)
{
    return (Dim){a.x + b.x, a.y + b.y, a.z + b.z};
}

static Dim
dim_sub(const Dim a, const Dim b)
{
    return (Dim){a.x - b.x, a.y - b.y, a.z - b.z};
}
*/

Block
blockCreateFromFile(const char* path, const Dim mm, const Dim offset, const Dim mm_sub)
{
    ERRCHK(mm_sub.x + offset.x <= mm.x);
    ERRCHK(mm_sub.y + offset.y <= mm.y);
    ERRCHK(mm_sub.z + offset.z <= mm.z);

    Block block = blockCreate(mm_sub);

    FILE* fp = fopen(path, "r");
    ERRCHK(fp);

    const Dim pitch = (Dim){
        mm.x - mm_sub.x,
        mm.y - mm_sub.y,
        mm.z - mm_sub.z,
    };

    const size_t base_offset = offset.x + offset.y * mm.x + offset.z * mm.x * mm.y;
    fseek(fp, base_offset, SEEK_SET);
    for (size_t k = 0; k < mm_sub.z; ++k) {
        for (size_t j = 0; j < mm_sub.y; ++j) {
            const size_t idx = j * mm_sub.x + k * mm_sub.x * mm_sub.y;
            const int res    = fread(&block.data[idx], sizeof(block.data[0]), mm_sub.x, fp);
            ERRCHK((size_t)res == mm_sub.x);

            ERRCHK(fseek(fp, sizeof(block.data[0]) * pitch.x, SEEK_CUR) == 0);
        }
        ERRCHK(fseek(fp, sizeof(block.data[0]) * pitch.x * pitch.y, SEEK_CUR) == 0);
    }

    fclose(fp);
    return block;
}

void
checkBlockContents(const char* path, const Block block)
{
    FILE* fp = fopen(path, "r");
    ERRCHK(fp);

    const size_t mm = block.mm.x * block.mm.y * block.mm.z;

    for (size_t i = 0; i < mm; ++i) {
        double val;
        fread(&val, sizeof(val), 1, fp);
        printf("%lu: Val %g, other %g\n", i, val, block.data[i]);
        if (val != block.data[i]) {
            for (size_t j = 0; i + j < mm; ++j)
                if (val == block.data[i + j])
                    printf("Found val %g at block index %lu\n", val, i + j);
        }
        ERRCHK(val == block.data[i]);
    }

    fclose(fp);
}

size_t
getIdx(const Dim spatial, const Block block)
{
    return spatial.x + spatial.y * block.mm.x + spatial.z * block.mm.x * block.mm.z;
}

void
interpolate(const Block in, const Dim halo_size, Block* out)
{
    const Dim start = halo_size;
    const Dim end   = (Dim){
        out->mm.x - halo_size.x,
        out->mm.y - halo_size.y,
        out->mm.z - halo_size.z,
    };

    const double mx_scale = (double)in.mm.x / out->mm.x;
    const double my_scale = (double)in.mm.y / out->mm.y;
    const double mz_scale = (double)in.mm.z / out->mm.z;

    for (size_t k0 = start.z; k0 < end.z; ++k0) {
        for (size_t j0 = start.y; j0 < end.y; ++j0) {
            for (size_t i0 = start.x; i0 < end.x; ++i0) {

                const double i = i0 + 0.5;
                const double j = j0 + 0.5;
                const double k = k0 + 0.5;

                const double xd = (i * mx_scale - floor(i * mx_scale)) /
                                  (ceil(i * mx_scale) - floor(i * mx_scale));
                const double yd = (j * my_scale - floor(j * my_scale)) /
                                  (ceil(j * my_scale) - floor(j * my_scale));
                const double zd = (k * mz_scale - floor(k * mz_scale)) /
                                  (ceil(k * mz_scale) - floor(k * mz_scale));

                const Dim c000 = (Dim){
                    (size_t)floor(i * mx_scale),
                    (size_t)floor(j * my_scale),
                    (size_t)floor(k * mz_scale),
                };

                const Dim c001 = (Dim){
                    (size_t)floor(i * mx_scale),
                    (size_t)floor(j * my_scale),
                    (size_t)ceil(k * mz_scale),
                };

                const Dim c010 = (Dim){
                    (size_t)floor(i * mx_scale),
                    (size_t)ceil(j * my_scale),
                    (size_t)floor(k * mz_scale),
                };

                const Dim c011 = (Dim){
                    (size_t)floor(i * mx_scale),
                    (size_t)ceil(j * my_scale),
                    (size_t)ceil(k * mz_scale),
                };

                const Dim c100 = (Dim){
                    (size_t)ceil(i * mx_scale),
                    (size_t)floor(j * my_scale),
                    (size_t)floor(k * mz_scale),
                };
                const Dim c101 = (Dim){
                    (size_t)ceil(i * mx_scale),
                    (size_t)floor(j * my_scale),
                    (size_t)ceil(k * mz_scale),
                };

                const Dim c110 = (Dim){
                    (size_t)ceil(i * mx_scale),
                    (size_t)ceil(j * my_scale),
                    (size_t)floor(k * mz_scale),
                };

                const Dim c111 = (Dim){
                    (size_t)ceil(i * mx_scale),
                    (size_t)ceil(j * my_scale),
                    (size_t)ceil(k * mz_scale),
                };

                const double c00 = in.data[getIdx(c000, in)] * (1.0 - xd) +
                                   in.data[getIdx(c100, in)] * xd;
                const double c01 = in.data[getIdx(c001, in)] * (1.0 - xd) +
                                   in.data[getIdx(c101, in)] * xd;
                const double c10 = in.data[getIdx(c010, in)] * (1.0 - xd) +
                                   in.data[getIdx(c110, in)] * xd;
                const double c11 = in.data[getIdx(c011, in)] * (1.0 - xd) +
                                   in.data[getIdx(c111, in)] * xd;

                const double c0 = c00 * (1.0 - yd) + c10 * yd;
                const double c1 = c01 * (1.0 - yd) + c11 * yd;

                const double c = c0 * (1.0 - zd) + c1 * zd;

                /*
                printDim("c000", c000);
                printDim("c001", c001);
                printDim("c010", c010);
                printDim("c011", c011);
                printDim("c100", c100);
                printDim("c101", c101);
                printDim("c110", c110);
                printDim("c111", c111);

                printf("Neighbors:\n");
                const Dim dims[] = {c000, c001, c010, c011, c100, c101, c110, c111};
                for (size_t w = 0; w < sizeof(dims) / sizeof(dims[0]); ++w)
                    printf("%lu: %g\n", w, in.data[getIdx(dims[w], in)]);
                printf("Interpolated value %g\n", c);
                // getchar();
                */
                const size_t out_idx = i0 + j0 * out->mm.x + k0 * out->mm.x * out->mm.y;
                out->data[out_idx]   = c;
            }
        }
    }
}

int
main(int argc, char* argv[])
{
    if (argc != 5) {
        fprintf(stderr, "Usage: ./pc-varfile-import <varfile> <mx> <my> <mz>\n");
        return EXIT_FAILURE;
    }

    const char* path       = argv[1];
    const Dim mm           = (Dim){atol(argv[2]), atol(argv[3]), atol(argv[4])};
    const size_t halo_size = 3;

    printf("Path: %s\n", path);

    const Dim mm_sub = mm;
    Block block      = blockCreateFromFile(path, mm, (Dim){0, 0, 0}, mm_sub);
    // checkBlockContents(path, block);
    Block block_out = blockCreate(
        (Dim){256 + 2 * halo_size, 256 + 2 * halo_size, 256 + 2 * halo_size});
    interpolate(block, (Dim){halo_size, halo_size, halo_size}, &block_out);
    printBlock("in", block);
    printBlock("out", block_out);

    blockDestroy(&block);
    blockDestroy(&block_out);

    return EXIT_SUCCESS;
}