#include "comm_data.h"

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

CommData
acCommDataCreate(const size_t ndims, const size_t* rr, const size_t* nn, const size_t nfields)
{
    // Determine the number of halo partitions
    const size_t npackets = powzu(3, ndims) - 1; // The neighbor count
    print("Creating npackets", npackets);

    // Create CommData
    CommData comm_data = (CommData){
        .npackets       = npackets,
        .local_packets  = malloc(sizeof(comm_data.local_packets[0]) * npackets),
        .remote_packets = malloc(sizeof(comm_data.remote_packets[0]) * npackets),
    };
    ERRCHK(comm_data.local_packets);
    ERRCHK(comm_data.remote_packets);

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
            ERRCHKK(curr <= npackets, "Packet counter OOB")
            comm_data.local_packets[curr]  = acCreatePackedData(ndims, dims, offset, nfields);
            comm_data.remote_packets[curr] = acCreatePackedData(ndims, dims, offset, nfields);
            ++curr;
        }
    }
    ERRCHKK(curr == npackets, "Did not created the expected number of packets");
    array_destroy(&segments);

    // for (size_t i = 0; i < npackets; ++i) {
    //     const size_t dims[]         = {3, 3, 3};
    //     const size_t offset[]       = {0, 0, 0};
    //     comm_data.local_packets[i]  = acCreatePackedData(ndims, dims, offset, nfields);
    //     comm_data.remote_packets[i] = acCreatePackedData(ndims, dims, offset, nfields);
    // }

    return comm_data;
}

void
acCommDataPrint(const char* label, const CommData comm_data)
{
    printf("CommData %s:\n", label);
    print("\tnpackets", comm_data.npackets);

    const size_t buflen = 128;
    char buf[buflen];

    for (size_t i = 0; i < comm_data.npackets; ++i) {
        snprintf(buf, buflen, "local_packets[%zu]", i);
        acPackedDataPrint(buf, comm_data.local_packets[i]);
    }

    for (size_t i = 0; i < comm_data.npackets; ++i) {
        snprintf(buf, buflen, "remote_packets[%zu]", i);
        acPackedDataPrint(buf, comm_data.remote_packets[i]);
    }
}

void
acCommDataDestroy(CommData* comm_data)
{
    for (size_t i = 0; i < comm_data->npackets; ++i) {
        acDestroyPackedData(&comm_data->local_packets[i]);
        acDestroyPackedData(&comm_data->remote_packets[i]);
    }
    free(comm_data->remote_packets);
    free(comm_data->local_packets);
    comm_data->npackets = 0;
}
