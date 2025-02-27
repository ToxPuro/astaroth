#include <cstdlib>
#include <vector>

#include "acm/detail/decomp.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/print_debug.h"

static void
test_get_nprocs_per_layer()
{
    {
        constexpr uint64_t    nprocs{64};
        std::vector<uint64_t> max_per_layer{2, 4};
        const auto nprocs_per_layer{ac::mpi::get_nprocs_per_layer(nprocs, max_per_layer)};
        ERRCHK(ac::mpi::prodd(nprocs_per_layer) == nprocs);
        PRINT_DEBUG_VECTOR(nprocs_per_layer);
    }
    {
        constexpr uint64_t    nprocs{64};
        std::vector<uint64_t> max_per_layer{2, 4, 4};
        const auto nprocs_per_layer{ac::mpi::get_nprocs_per_layer(nprocs, max_per_layer)};
        ERRCHK(ac::mpi::prodd(nprocs_per_layer) == nprocs);
        PRINT_DEBUG_VECTOR(nprocs_per_layer);
    }
    {
        constexpr uint64_t    nprocs{2};
        std::vector<uint64_t> max_per_layer{8, 4};
        const auto nprocs_per_layer{ac::mpi::get_nprocs_per_layer(nprocs, max_per_layer)};
        ERRCHK(ac::mpi::prodd(nprocs_per_layer) == nprocs);
        PRINT_DEBUG_VECTOR(nprocs_per_layer);
    }
    {
        constexpr uint64_t    nprocs{8};
        std::vector<uint64_t> max_per_layer{4, 4};
        const auto nprocs_per_layer{ac::mpi::get_nprocs_per_layer(nprocs, max_per_layer)};
        ERRCHK(ac::mpi::prodd(nprocs_per_layer) == nprocs);
        PRINT_DEBUG_VECTOR(nprocs_per_layer);
    }
    {
        constexpr uint64_t    nprocs{64};
        std::vector<uint64_t> max_per_layer{2, 2, 2, 4, 4};
        const auto nprocs_per_layer{ac::mpi::get_nprocs_per_layer(nprocs, max_per_layer)};
        ERRCHK(ac::mpi::prodd(nprocs_per_layer) == nprocs);
        PRINT_DEBUG_VECTOR(nprocs_per_layer);
    }
    {
        constexpr uint64_t    nprocs{64};
        std::vector<uint64_t> max_nprocs_per_layer{2, 4};
        const auto nprocs_per_layer{ac::mpi::get_nprocs_per_layer(nprocs, max_nprocs_per_layer)};
        const ac::shape global_nn{128, 128, 128};
        auto            decomp{decompose_hierarchical(global_nn, nprocs_per_layer)};

        const auto global_decomp{hierarchical_decomposition_to_global(decomp)};
        PRINT_DEBUG(global_decomp);
        PRINT_DEBUG(decomp);
        ERRCHK((global_decomp == ac::shape{4, 4, 4}));
        ERRCHK(prod(global_decomp) == nprocs);
    }
}

int
main()
{
    test_get_nprocs_per_layer();

    // TODO proper test (but seems to work)
    // ac::mpi::init_funneled();
    // const ac::shape global_nn{256, 128, 64};
    // MPI_Comm cart_comm{ac::mpi::cart_comm_hierarchical_create(MPI_COMM_WORLD, global_nn)};
    // ac::mpi::cart_comm_destroy(&cart_comm);
    // ac::mpi::finalize();

    // std::cout << "-----" << std::endl;
    // test::print(1);
    // std::cout << "-----" << std::endl;
    // test::print(std::vector{1, 2, 3});
    // std::cout << "-----" << std::endl;
    // test::print(std::vector{std::vector{1, 2, 3}, std::vector{4, 5, 6}});
    // std::cout << "-----" << std::endl;
    // test::print_debug("lala",
    //                   std::vector{std::vector{std::vector{1, 2, 3}, std::vector{4, 5, 6}},
    //                               std::vector{std::vector{1, 2, 3}, std::vector{4, 5, 6}}});
    // std::cout << "-----" << std::endl;
    // test::print_debug("lala",
    //                   std::vector{std::vector{std::vector{std::vector{1, 2, 3},
    //                                                       std::vector{4, 5, 6}},
    //                                           std::vector{std::vector{1, 2, 3},
    //                                                       std::vector{4, 5, 6}}},
    //                               std::vector{std::vector{std::vector{1, 2, 3},
    //                                                       std::vector{4, 5, 6}},
    //                                           std::vector{std::vector{1, 2, 3},
    //                                                       std::vector{4, 5, 6}}}});
    // std::cout << "-----" << std::endl;

    /*
    Label: { // indent 0
        { // indent 1
            { 1 2 3 } // Indent 2
        } // indent 1
    }// indent 0

    indent after each endl
    */
    PRINT_LOG_INFO("OK");
    return EXIT_SUCCESS;
}
