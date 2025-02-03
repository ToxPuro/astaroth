#include "revised.h"

#include <iostream>

#include "errchk.h"
#include "memory_resource.h"

namespace ac {

template <typename T, typename MemoryResource> class pointer {
  private:
    size_t m_count{0};
    T* m_data{nullptr};

  public:
    pointer(const size_t count, const T* data)
        : m_count{count}, m_data{data}
    {
    }

    auto size() const { return m_count; }

    auto data() const { return m_data; }
    auto data() { return m_data; }

    auto begin() const { return data(); }
    auto begin() { return data(); }

    auto end() const { return data() + size(); }
    auto end() { return data() + size(); }

    T& operator[](const size_t i)
    {
        ERRCHK(i < size());
        return m_data[i];
    }
    const T& operator[](const size_t i) const
    {
        ERRCHK(i < size());
        return m_data[i];
    }
};

template <typename T> using host_pointer   = pointer<T, ac::mr::host_memory_resource>;
template <typename T> using device_pointer = pointer<T, ac::mr::device_memory_resource>;

} // namespace ac

void
test_acm_revised()
{
    std::cout << "hello from revised" << std::endl;
}
