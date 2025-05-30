#pragma once
#include <future>

namespace ac {

template <typename T> class future {
  private:
    std::future<T> m_future;

  public:
    template <typename Function, typename... Args>
    future(Function&& fn, Args&&... args)
        : m_future{std::async(std::launch::async, fn, args...)}
    {
    }

    bool complete() const noexcept { return m_future.valid(); }
    bool ready() const noexcept { return m_future.wait_for(0) == std::future_status::ready; }

    auto wait()
    {
        ERRCHK(m_future.valid());
        return m_future.get();
    }
};

} // namespace ac
