#pragma once
#include <chrono>
#include <fstream>
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

    auto lap_ns()
    {
        const auto ns_elapsed{diff_ns()};
        reset();
        return ns_elapsed;
    }

    void print_lap(const std::string& label = "Elapsed")
    {
        std::cout << label << ":\t" << diff_ns() << " ns" << std::endl;
        reset();
    }
};

class timewriter {
  private:
    ac::timer   m_timer;
    std::string m_path;

  public:
    timewriter(const std::string& path)
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
        file << label << "," << m_timer.lap_ns() << std::endl;
        file.close();
    }
};

} // namespace ac
