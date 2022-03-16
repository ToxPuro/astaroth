#include <stdbool.h>
#include <stdint.h>
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
dimPrint(const char* id, const Dim dim)
{
    printf("%s: (%lu, %lu, %lu)\n", id, dim.x, dim.y, dim.z);
}

typedef struct {
    Dim nn;
    Dim mm;
    size_t pad;
    real* data;
} Block;

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
printNelems(const real* data, const Dim mm, const Dim offset, const size_t nelems)
{
    ERRCHK(offset.x + nelems < mm.x);
    for (size_t i = 0; i < nelems; ++i) {
        printf("%g, ", data[i + offset.x + offset.y * mm.x + offset.z * mm.x * mm.y]);
    }
}

void
blockPrint(const char* id, const Block block)
{
    printf("Block `%s`\n", id);
    dimPrint("\tnn", block.nn);
    dimPrint("\tmm", block.mm);
    printf("\tpad: %lu\n", block.pad);
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
blockCreate(const Dim nn, const size_t pad)
{
    const Dim mm = (Dim){
        nn.x + 2 * pad,
        nn.y + 2 * pad,
        nn.z + 2 * pad,
    };

    Block block = {
        .nn   = nn,
        .mm   = mm,
        .pad  = pad,
        .data = NULL,
    };
    block.data = malloc(sizeof(block.data[0]) * mm.x * mm.y * mm.z);
    ERRCHK(block.data);

    return block;
}

void
blockDestroy(Block* block)
{
    block->mm = block->nn = (Dim){0, 0, 0};
    free(block->data);
    block->data = NULL;
}

Block
blockCreateFromFile(const char* path, const Dim offset, const Dim nn, const size_t pad,
                    const Dim nn_sub)
{
    const Dim mm = (Dim){nn.x + 2 * pad, nn.y + 2 * pad, nn.z + 2 * pad};

    Block block      = blockCreate(nn_sub, pad);
    const Dim mm_sub = block.mm;

    FILE* fp = fopen(path, "r");
    ERRCHK(fp);

    const size_t base_offset = offset.x + offset.y * mm.x + offset.z * mm.x * mm.y;
    fseek(fp, base_offset, SEEK_SET);
    for (size_t k = 0; k < mm_sub.z; ++k) {
        for (size_t j = 0; j < mm_sub.y; ++j) {
            const size_t idx = j * mm_sub.x + k * mm_sub.x * mm_sub.y;
            const int res    = fread(&block.data[idx], sizeof(block.data[0]), mm_sub.x, fp);
            ERRCHK((size_t)res == mm_sub.x);

            ERRCHK(fseek(fp, sizeof(block.data[0]) * mm.x, SEEK_CUR) == 0);
        }
        ERRCHK(fseek(fp, sizeof(block.data[0]) * mm.x * mm.y, SEEK_CUR) == 0);
    }

    fclose(fp);

    return block;
}

int
main(int argc, char* argv[])
{
    if (argc != 6) {
        fprintf(stderr, "Usage: ./pc-varfile-import <varfile> <nx> <ny> <nz> <bound>\n");
        return EXIT_FAILURE;
    }

    const char* path = argv[1];
    const Dim nn     = (Dim){atol(argv[2]), atol(argv[3]), atol(argv[4])};
    const size_t pad = atol(argv[5]);

    printf("Path: %s\n", path);
    dimPrint("Dims", nn);

    const Dim nn_sub = (Dim){nn.x / 4, nn.y / 4, nn.z / 4};
    Block block      = blockCreateFromFile(path, (Dim){0, 0, 0}, nn, pad, nn_sub);
    blockPrint("test", block);
    blockDestroy(&block);
    /*
    FILE* fp = fopen(path, "r");
    ERRCHK(fp);

    while (true) {
        double val;
        const int res = fread(&val, sizeof(val), 1, fp);
        if (res != 1) {
            if (feof(fp))
                break;
            else
                ERROR("Read error");
        }
        else {
            // printf("%lg\n", val);
        }
    }

    fclose(fp);
    */

    return EXIT_SUCCESS;
}