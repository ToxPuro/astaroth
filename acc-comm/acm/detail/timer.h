#pragma once
#include <chrono>
#include <iostream>

namespace ac {
class timer {
  private:
    using clock      = std::chrono::steady_clock;
    using time_point = std::chrono::time_point<clock>;

    time_point m_start;

  public:
    timer()
        : m_start{clock::now()}
    {
    }

    void reset() { m_start = clock::now(); }
    auto diff() { return clock::now() - m_start; }
    auto diff_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(diff()).count(); }

    void lap(const std::string& label = "Elapsed")
    {
        std::cout << label << ":\t" << diff_ns() << " ns" << std::endl;
        reset();
    }
};
} // namespace ac
