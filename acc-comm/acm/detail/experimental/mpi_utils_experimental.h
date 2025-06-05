#pragma once
#include <functional>

#include "acm/detail/buffer.h"
#include "acm/detail/errchk_mpi.h"
#include "acm/detail/mpi_utils.h"
#include "acm/detail/ntuple.h"
#include "acm/detail/view.h"

// These should be merged to mpi_utils.h eventually

namespace ac::mpi {

class comm {
  private:
    std::unique_ptr<MPI_Comm, std::function<void(MPI_Comm*)>> m_comm;

  public:
    comm()
        : m_comm{new MPI_Comm{MPI_COMM_NULL}, [](MPI_Comm* ptr) {
                     if (*ptr != MPI_COMM_NULL)
                         ERRCHK_MPI_API(MPI_Comm_free(ptr));
                     delete ptr;
                 }}
    {
    }

    explicit comm(const MPI_Comm& parent_comm, const bool take_ownership = false)
        : comm{}
    {
        if (take_ownership)
            *m_comm = parent_comm;
        else
            ERRCHK_MPI_API(MPI_Comm_dup(parent_comm, m_comm.get()));
    }

    const MPI_Comm& get() const { return *m_comm; }
};

class cart_comm {
  private:
    comm      m_comm;
    ac::shape m_global_nn;

  public:
    cart_comm(
        const MPI_Comm& parent_comm, const ac::shape& global_nn,
        const ac::mpi::RankReorderMethod& reorder_method = ac::mpi::RankReorderMethod::hierarchical)
        : m_comm{ac::mpi::cart_comm_create(parent_comm, global_nn, reorder_method), true},
          m_global_nn{global_nn}
    {
    }

    const MPI_Comm& get() const { return m_comm.get(); }
    ac::shape       global_nn() const { return m_global_nn; }
};

class datatype {
  private:
    std::unique_ptr<MPI_Datatype, std::function<void(MPI_Datatype*)>> m_datatype;

  public:
    datatype()
        : m_datatype{new MPI_Datatype{MPI_DATATYPE_NULL}, [](MPI_Datatype* ptr) {
                         ERRCHK_MPI(ptr != nullptr);
                         if (*ptr != MPI_DATATYPE_NULL)
                             ERRCHK_MPI_API(MPI_Type_free(ptr));
                         delete ptr;
                     }}
    {
    }

    explicit datatype(const MPI_Datatype& parent_datatype, const bool take_ownership = false)
        : datatype{}
    {
        if (take_ownership)
            *m_datatype = parent_datatype;
        else
            ERRCHK_MPI_API(MPI_Type_dup(parent_datatype, m_datatype.get()));
    }

    const MPI_Datatype& get() const { return *m_datatype; }
};

template <typename T> class subarray {
  private:
    datatype m_datatype;

  public:
    subarray(const ac::shape& dims, const ac::shape& subdims, const ac::index& offset)
        : m_datatype{ac::mpi::subarray_create(dims, subdims, offset, ac::mpi::get_dtype<T>()), true}
    {
    }

    const MPI_Datatype& get() const { return m_datatype.get(); }
};

template <typename T> class hindexed_block {
  private:
    datatype m_datatype;

  public:
    hindexed_block() = default;

    // Must use MPI_BOTTOM for the addresses to be treated as absolute.
    // Otherwise, the data would have to come from a single, contiguous allocation
    hindexed_block(const ac::shape& dims, const ac::shape& subdims, const ac::index& offset,
                   const std::vector<T*>& pointers)
    {
        const int nblocks{as<int>(pointers.size())};
        const int block_length{1};
        auto      displacements{ac::make_ntuple<MPI_Aint>(pointers.size(), 0)};
        for (size_t i{0}; i < pointers.size(); ++i)
            ERRCHK_MPI_API(MPI_Get_address(pointers[i], &displacements[i]));

        const mpi::subarray<T> block{dims, subdims, offset};

        MPI_Datatype hindexed_block_dtype{MPI_DATATYPE_NULL};
        ERRCHK_MPI_API(MPI_Type_create_hindexed_block(nblocks,
                                                      block_length,
                                                      displacements.data(),
                                                      block.get(),
                                                      &hindexed_block_dtype));
        ERRCHK_MPI_API(MPI_Type_commit(&hindexed_block_dtype));
        m_datatype = datatype{hindexed_block_dtype, true};
    }

    const MPI_Datatype& get() const { return m_datatype.get(); }
};

class request {
  private:
    std::unique_ptr<MPI_Request, std::function<void(MPI_Request*)>> m_req;

  public:
    /**
     * Wrap an MPI_Request.
     * Does not take ownership, starts only to track the resource and raises an error
     * if request goes out of scope without being released/waited upon.
     */
    explicit request(const MPI_Request& req = MPI_REQUEST_NULL)
        : m_req{new MPI_Request{req}, [](MPI_Request* ptr) {
                    ERRCHK_MPI_EXPR_DESC(*ptr == MPI_REQUEST_NULL,
                                         "Attempted to destruct a request still in flight. Ensure "
                                         "wait is called properly before leaving scope.");
                    delete ptr;
                }}
    {
    }

    MPI_Request* data() const noexcept
    {
        ERRCHK_MPI(complete());
        return m_req.get();
    }

    bool complete() const noexcept { return *m_req == MPI_REQUEST_NULL; }

    bool ready() const noexcept
    {
        ERRCHK_MPI(!complete());
        int flag;
        ERRCHK_MPI_API(MPI_Request_get_status(*m_req, &flag, MPI_STATUS_IGNORE));
        return flag;
    }

    void wait() noexcept
    {
        ERRCHK_MPI(!complete());
        ERRCHK_MPI_API(MPI_Wait(m_req.get(), MPI_STATUS_IGNORE));
        ERRCHK_MPI(complete());
    }
};

template <typename T>
[[nodiscard]] auto
isend(const MPI_Comm& comm, const int16_t tag, const ac::base_view<T>& in, const int dst)
{
    ac::mpi::request req;
    ERRCHK_MPI_API(MPI_Isend(in.data(),
                             as<int>(in.size()),
                             ac::mpi::get_dtype<T>(),
                             dst,
                             tag,
                             comm,
                             req.data()));
    return req;
}

template <typename T>
[[nodiscard]] auto
irecv(const MPI_Comm& comm, const int16_t tag, const int src, ac::base_view<T> out)
{
    ac::mpi::request req;
    ERRCHK_MPI_API(MPI_Irecv(out.data(),
                             as<int>(out.size()),
                             ac::mpi::get_dtype<T>(),
                             src,
                             tag,
                             comm,
                             req.data()));
    return req;
}

template <typename T, typename U>
[[nodiscard]] auto
iallreduce(const MPI_Comm& comm, const ac::base_view<T>& in, const MPI_Op& op, ac::base_view<U> out)
{
    static_assert(std::is_same_v<std::remove_const_t<T>, U>);
    ERRCHK_MPI(in.size() <= out.size());

    ac::mpi::request req;
    ERRCHK_MPI_API(MPI_Iallreduce(in.data(),
                                  out.data(),
                                  as<int>(in.size()),
                                  ac::mpi::get_dtype<T>(),
                                  op,
                                  comm,
                                  req.data()));
    return req;
}

template <typename T, typename Allocator> class buffered_isend {
  private:
    ac::buffer<T, Allocator> m_buf;
    ac::mpi::request         m_req{MPI_REQUEST_NULL};

  public:
    buffered_isend(const size_t max_count)
        : m_buf{max_count}
    {
    }

    void launch(const MPI_Comm& comm, const int16_t tag, const ac::view<T, Allocator>& in,
                const int dst)
    {
        ERRCHK_MPI(m_req.complete());
        ac::copy(in, m_buf.get());
        m_req = ac::mpi::isend(comm, tag, ac::make_view(in.size(), 0, m_buf.get()), dst);
    }

    void wait() { m_req.wait(); }
};

/**
 * Buffered asynchronous allreduce operation with dynamic resizing.
 * Note: never shrinks
 */
template <typename T, typename Allocator> class buffered_iallreduce {
  private:
    ac::buffer<T, Allocator> m_buf{0};
    ac::mpi::request         m_req{MPI_REQUEST_NULL};

  public:
    buffered_iallreduce() = default;

    template <typename U>
    void launch(const MPI_Comm& comm, const ac::view<T, Allocator>& in, const MPI_Op& op,
                ac::base_view<U> out)
    {
        ERRCHK_MPI(m_req.complete());

        // Resize buffer if needed
        if (m_buf.size() < in.size())
            m_buf = ac::buffer<T, Allocator>{in.size()};

        ac::copy(in, m_buf.get());
        m_req = ac::mpi::iallreduce(comm, ac::make_view(in.size(), 0, m_buf.get()), op, out);
    }

    void wait() { m_req.wait(); }
};

template <typename T, typename Allocator> class twoway_buffered_iallreduce {
  private:
    ac::buffer<T, Allocator>                   m_buf{0};
    ac::mpi::buffered_iallreduce<T, Allocator> m_buffered_iallreduce{};

  public:
    twoway_buffered_iallreduce() = default;

    void launch(const MPI_Comm& comm, const ac::view<T, Allocator>& in, const MPI_Op& op)
    {
        // Resize buffer if needed
        if (m_buf.size() < in.size())
            m_buf = ac::buffer<T, Allocator>{in.size()};

        m_buffered_iallreduce.launch(comm, in, op, m_buf.get());
    }

    void wait(ac::view<T, Allocator> out)
    {
        m_buffered_iallreduce.wait();

        ERRCHK_MPI(out.size() <= m_buf.size());
        ac::copy(ac::make_view(out.size(), 0, m_buf.get()), out);
    }
};

} // namespace ac::mpi

// Further helper functions

namespace ac::mpi {

ac::shape global_mm(const cart_comm& comm, const ac::index& rr);

ac::shape global_nn(const cart_comm& comm);

ac::shape local_mm(const cart_comm& comm, const ac::index& rr);

ac::shape local_nn(const cart_comm& comm);

/** Returns the coordinates of processes w.r.t. their MPI_COMM_WORLD ranks */
std::vector<ac::index> get_rank_ordering(const MPI_Comm& cart_comm);

} // namespace ac::mpi

namespace ac::mpi {

// static ac::mpi::cart_comm
// make_cart_comm(const MPI_Comm& parent_comm, const ac::shape& global_nn)
// {
//     // Get the number of processes
//     int mpi_nprocs{-1};
//     ERRCHK_MPI_API(MPI_Comm_size(parent_comm, &mpi_nprocs));

//     // Get ndims
//     const size_t ndims{global_nn.size()};

//     // Decompose all dimensions
//     ac::mpi::shape mpi_decomp{ac::make_ntuple<int>(ndims, 0)};
//     ERRCHK_MPI_API(MPI_Dims_create(mpi_nprocs, as<int>(ndims), mpi_decomp.data()));

//     // Decompose only the slowest moving dimension (last dimension in Astaroth, first in MPI)
//     // mpi::shape mpi_decomp(ndims, 1);
//     // mpi_decomp[0] = 0;
//     // ERRCHK_MPI_API(MPI_Dims_create(mpi_nprocs, as<int>(ndims), mpi_decomp.data()));

//     // Create the Cartesian communicator
//     MPI_Comm   cart_comm{MPI_COMM_NULL};
//     mpi::shape mpi_periods{ac::make_ntuple<int>(ndims, 1)}; // Periodic in all dimensions
//     // int reorder{1}; // Enable reordering (but likely inop with most MPI implementations)
//     ERRCHK_MPI_API(MPI_Cart_create(parent_comm,
//                                    as<int>(ndims),
//                                    mpi_decomp.data(),
//                                    mpi_periods.data(),
//                                    reorder,
//                                    &cart_comm));

//     // Can also add custom decomposition and rank reordering here instead:
//     // int reorder{0};
//     // ...
//     return cart_comm;
// }

} // namespace ac::mpi

namespace ac::mpi {

/** Select the device and return its id */
int select_device_lumi();

} // namespace ac::mpi

namespace ac::mpi {

uint64_t rank(const ac::mpi::comm& comm);
uint64_t size(const ac::mpi::comm& comm);

} // namespace ac::mpi
