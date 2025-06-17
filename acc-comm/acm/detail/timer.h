#pragma once
#include <chrono>
#include <fstream>
#include <iostream>

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
    ac::timer   m_timer;
    std::string m_path;

  public:
    explicit timewriter(const std::string& path)
        : m_path{path}
    {
        std::ofstream file;
        file.open(m_path);
        file << "label,ns" << std::endl;
        file.close();
    }

    void log(const std::string& label)
    {
        std::ofstream file;
        file.open(m_path, std::ios_base::app);
        file << label << ","
             << std::chrono::duration_cast<std::chrono::nanoseconds>(m_timer.lap()).count()
             << std::endl;
        file.close();
    }
};

} // namespace ac
