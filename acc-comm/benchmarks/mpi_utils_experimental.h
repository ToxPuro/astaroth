#pragma once
#include <functional>

#include "acm/detail/errchk_mpi.h"
#include "acm/detail/mpi_utils.h"

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

        MPI_Datatype hindexed_block{MPI_DATATYPE_NULL};
        ERRCHK_MPI_API(MPI_Type_create_hindexed_block(nblocks,
                                                      block_length,
                                                      displacements.data(),
                                                      block.get(),
                                                      &hindexed_block));
        ERRCHK_MPI_API(MPI_Type_commit(&hindexed_block));
        m_datatype = datatype{hindexed_block, true};
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
                    ERRCHK_MPI(*ptr == MPI_REQUEST_NULL);
                    delete ptr;
                }}
    {
    }

    void wait() noexcept
    {
        ERRCHK_MPI(*m_req != MPI_REQUEST_NULL);
        ERRCHK_MPI_API(MPI_Wait(m_req.get(), MPI_STATUS_IGNORE));
        ERRCHK_MPI(*m_req == MPI_REQUEST_NULL);
    }
};

} // namespace ac::mpi

// Further helper functions

namespace ac::mpi {

ac::shape
global_mm(const cart_comm& comm, const ac::index& rr)
{
    return ac::mpi::get_global_mm(comm.global_nn(), rr);
}
ac::shape
global_nn(const cart_comm& comm)
{
    return comm.global_nn();
}

ac::shape
local_mm(const cart_comm& comm, const ac::index& rr)
{
    return ac::mpi::get_local_mm(comm.get(), comm.global_nn(), rr);
}
ac::shape
local_nn(const cart_comm& comm)
{
    return ac::mpi::get_local_nn(comm.get(), comm.global_nn());
}

} // namespace ac::mpi
