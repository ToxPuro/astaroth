#include "halo_segment_batch.h"

#include "dynamic_array.h"
#include "errchk.h"
#include "math.h"
#include "math_utils.h"
#include "ndarray.h"
#include "print.h"

/** Writes the combinations to the output paramete and returns the number of combinations */
size_t
recurse_combinations(const size_t start, const size_t ndims, const size_t* combination,
                     DynamicArray* combinations)
{
    ERRCHK(ndims > 0);

    size_t counter = 1;
    for (size_t i = start; i < ndims; ++i) {
        size_t new_combination[ndims];
        copy(ndims, combination, new_combination);
        new_combination[i] = 1;
        counter += recurse_combinations(i + 1, ndims, new_combination, combinations);
    }
    array_append_multiple(ndims, combination, combinations);
    return counter;
}

DynamicArray
create_combinations(const size_t ndims)
{
    const size_t ncombinations = count_combinations(ndims);
    const size_t count         = ndims * ncombinations;
    DynamicArray combinations  = array_create(count);

    size_t initial_combination[ndims];
    set(0, ndims, initial_combination);
    size_t counter = recurse_combinations(0, ndims, initial_combination, &combinations);
    ERRCHK(ncombinations == counter);

    // print_ndarray("Combinations", 2, (size_t[]){ndims, ncombinations}, combinations.data);

    return combinations;
}

static void
get_mm(const size_t ndims, const size_t* nn, const size_t* rr, size_t* mm)
{
    for (size_t i = 0; i < ndims; ++i)
        mm[i] = 2 * rr[i] + nn[i];
}

static void
get_nn(const size_t ndims, const size_t* mm, const size_t* rr, size_t* nn)
{
    for (size_t i = 0; i < ndims; ++i)
        nn[i] = mm[i] - 2 * rr[i];
}

void
acHaloSegmentBatchTest(const size_t ndims, const size_t* nn, const size_t* rr,
                       const HaloSegmentBatch batch)
{
    // Ensure that
    // 1. none of the segments overlap
    // 2. the number of points covered by the segments encompass the whole
    //    ghost zone
    // 3. all of the segments are within mm

    { // Local packets
        size_t mm[ndims];
        get_mm(ndims, nn, rr, mm);
        const size_t model_count = prod(ndims, mm) - prod(ndims, nn);

        size_t count = 0;
        for (size_t i = 0; i < batch.npackets; ++i) {
            const HaloSegment a = batch.local_packets[i];
            ERRCHK(ndims == a.ndims);
            for (size_t j = i + 1; j < batch.npackets; ++j) {
                const HaloSegment b = batch.local_packets[j];

                ERRCHK(intersect_box(ndims, a.offset, a.dims, b.offset, b.dims) == false);
            }
            count += prod(ndims, a.dims);

            for (size_t j = 0; j < ndims; ++j)
                ERRCHK(a.offset[j] + a.dims[j] <= mm[j]);
        }
        ERRCHK(count == model_count);
    }
    { // Remote packets
        size_t mm[ndims];
        get_mm(ndims, nn, rr, mm);
        const size_t model_count = prod(ndims, mm) - prod(ndims, nn);

        size_t count = 0;
        for (size_t i = 0; i < batch.npackets; ++i) {
            const HaloSegment a = batch.remote_packets[i];
            ERRCHK(ndims == a.ndims);
            for (size_t j = i + 1; j < batch.npackets; ++j) {
                const HaloSegment b = batch.remote_packets[j];

                ERRCHK(intersect_box(ndims, a.offset, a.dims, b.offset, b.dims) == false);
            }
            count += prod(ndims, a.dims);

            for (size_t j = 0; j < ndims; ++j)
                ERRCHK(a.offset[j] + a.dims[j] <= mm[j]);
        }
        ERRCHK(count == model_count);
    }
}

HaloSegmentBatch
acHaloSegmentBatchCreate(const size_t ndims, const size_t* nn, const size_t* rr,
                         const size_t nfields)
{
    // Determine the number of halo partitions
    const size_t npackets = powzu(3, ndims) - 1; // The neighbor count
    print("Creating npackets", npackets);

    // Create HaloSegmentBatch
    HaloSegmentBatch batch = (HaloSegmentBatch){
        .npackets       = npackets,
        .local_packets  = malloc(sizeof(batch.local_packets[0]) * npackets),
        .remote_packets = malloc(sizeof(batch.remote_packets[0]) * npackets),
    };
    ERRCHK(batch.local_packets);
    ERRCHK(batch.remote_packets);

    DynamicArray segments  = create_combinations(ndims);
    const size_t nsegments = segments.len / ndims;
    // print_ndarray("Segments", 2, (size_t[]){ndims, nsegments}, segments.data);

    size_t curr = 0;
    for (size_t i = 0; i < nsegments; ++i) {
        // Get the current segment
        const size_t* segment = &segments.data[i * ndims];

        // Skip the segment containing the whole computational domain
        if (popcount(ndims, segment) == 0)
            continue;

        // Get the dimensions of the segment
        size_t dims[ndims];
        for (size_t j = 0; j < ndims; ++j)
            dims[j] = segment[j] * rr[j] + (1 - segment[j]) * nn[j];

        // print_array("Current segment", ndims, segment);
        // print_array("Current dims", ndims, dims);

        // Get the offsets of the segment
        size_t current_repeated[ndims * nsegments];
        repeat(ndims, segment, nsegments, current_repeated);
        // print_ndarray("Current", 2, (size_t[]){ndims, nsegments}, current_repeated);

        size_t hadamard[ndims * nsegments];
        mul(ndims * nsegments, current_repeated, segments.data, hadamard);
        // print_ndarray("Hadamard", 2, (size_t[]){ndims, nsegments}, hadamard);

        size_t combinations[ndims * nsegments];
        const size_t combinations_len = unique_subsets(ndims * nsegments, hadamard, ndims,
                                                       combinations);
        const size_t ncombinations    = combinations_len / ndims;
        // print_ndarray("combinations", 2, (size_t[]){ndims, nrows}, combinations);

        for (size_t j = 0; j < ncombinations; ++j) {
            const size_t* combination = &combinations[j * ndims];

            size_t offset[ndims];
            for (size_t k = 0; k < ndims; ++k)
                offset[k] = combination[k] * (rr[k] + nn[k]) +
                            (1 - combination[k]) * (1 - segment[k]) * rr[k];
            // print_array("Current projected offset", ndims, offset);
            ERRCHKK(curr <= npackets, "Packet counter OOB");
            batch.local_packets[curr]  = acCreateHaloSegment(ndims, dims, offset, nfields);
            batch.remote_packets[curr] = acCreateHaloSegment(ndims, dims, offset, nfields);
            // print_array("Local packet offset offset", ndims,
            // batch.local_packets[curr].offset);
            ++curr;
        }
    }
    ERRCHKK(curr == npackets, "Did not created the expected number of packets");
    acHaloSegmentBatchTest(ndims, nn, rr, batch);
    array_destroy(&segments);

    // for (size_t i = 0; i < npackets; ++i) {
    //     const size_t dims[]         = {3, 3, 3};
    //     const size_t offset[]       = {0, 0, 0};
    //     batch.local_packets[i]  = acCreateHaloSegment(ndims, dims, offset, nfields);
    //     batch.remote_packets[i] = acCreateHaloSegment(ndims, dims, offset, nfields);
    // }

    return batch;
}

void
acHaloSegmentBatchPrint(const char* label, const HaloSegmentBatch batch)
{
    printf("HaloSegmentBatch %s:\n", label);
    print("\tnpackets", batch.npackets);

    const size_t buflen = 128;
    char buf[buflen];

    for (size_t i = 0; i < batch.npackets; ++i) {
        snprintf(buf, buflen, "local_packets[%zu]", i);
        acHaloSegmentPrint(buf, batch.local_packets[i]);
    }

    for (size_t i = 0; i < batch.npackets; ++i) {
        snprintf(buf, buflen, "remote_packets[%zu]", i);
        acHaloSegmentPrint(buf, batch.remote_packets[i]);
    }
}

void
acHaloSegmentBatchDestroy(HaloSegmentBatch* batch)
{
    for (size_t i = 0; i < batch->npackets; ++i) {
        acDestroyHaloSegment(&batch->local_packets[i]);
        acDestroyHaloSegment(&batch->remote_packets[i]);
    }
    free(batch->remote_packets);
    free(batch->local_packets);
    batch->npackets = 0;
}
