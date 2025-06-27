#pragma once
#include <chrono>
#include <fstream>
#include <iostream>

#include "acm/detail/errchk.h"

namespace ac {
class timer {
  private:
    using clock      = std::chrono::steady_clock;
    using time_point = clock::time_point;
    using duration   = clock::duration;

    time_point m_start{clock::now()};

  public:
    duration diff() const { return clock::now() - m_start; }
    void     reset() { m_start = clock::now(); }
    duration lap()
    {
        const auto elapsed{diff()};
        reset();
        return elapsed;
    }
};

class timewriter {
  private:
    ac::timer     m_timer;
    std::ofstream m_file;

  public:
    explicit timewriter(const std::string& path)
        : m_file{path}
    {
        ERRCHK(m_file);
        ERRCHK(m_file << "label,ns" << std::endl);
    }

    void log(const std::string& label)
    {
        ERRCHK(m_file << label << ","
                      << std::chrono::duration_cast<std::chrono::nanoseconds>(m_timer.lap()).count()
                      << std::endl);
    }
};

} // namespace ac
