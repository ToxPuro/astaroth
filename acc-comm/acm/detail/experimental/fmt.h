#pragma once
#include <iomanip>
#include <iostream>
#include <sstream>

#include "acm/detail/errchk.h"

namespace ac::fmt {

constexpr auto delimiter{','};

struct lossless {
    static void configure(std::ostream& os) { ERRCHK(os << std::hexfloat); }
};

struct human_readable {
    static void configure(std::ostream& os)
    {
        ERRCHK(os << std::defaultfloat << std::setprecision(3));
    }
};

template <typename Formatter = ac::fmt::lossless, typename T, typename... Args>
auto
push(std::ostream& stream, T&& first, Args&&... args)
{
    Formatter::configure(stream);
    ERRCHK(stream << std::forward<T>(first));
    ((stream << delimiter << std::forward<Args>(args)), ...);
    ERRCHK(stream << std::endl);
}

template <typename T>
auto
pull_token(std::istream& is, T& output)
{
    // Fetch token
    std::string token;
    ERRCHK(std::getline(is, token, delimiter));

    // Parse token
    std::istringstream iss{token};
    ERRCHK(iss);
    //ERRCHK(iss >> output); // Reading from std::hexfloat bugged on Mahti
    if constexpr (std::is_same_v<double, T>) {
	    output = std::strtod(iss.str().c_str(), nullptr);
	    ERRCHK(output != 0);
    } else if constexpr(std::is_same_v<float, T>) {
	    output = std::strtof(iss.str().c_str(), nullptr);
	    ERRCHK(output != 0);
    } else {
	    ERRCHK(iss >> output);
    }
    ERRCHK(!iss.fail());
}

template <typename... Args>
auto
pull(std::istream& stream, Args&&... args)
{
    ERRCHK(!stream.fail());
    (pull_token(stream, std::forward<Args>(args)), ...);
    ERRCHK(!stream.fail());

    // Check that there are no unparsed tokens
    std::string remaining;
    ERRCHK(!std::getline(stream, remaining, delimiter));
    ERRCHK(stream.fail() && !stream.bad()); // Confirm this was erroneous but not catastrophically
    stream.clear();                         // Clear the error
}

} // namespace ac::fmt
