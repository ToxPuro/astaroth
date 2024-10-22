#pragma once
#include <iostream>
#include <stdexcept>

#include "errchk_print.h"

#define ERRCHK(expr)                                                                               \
    do {                                                                                           \
        if (!(expr)) {                                                                             \
            errchk_print_error(__func__, __FILE__, __LINE__, #expr, "");                           \
            errchk_print_stacktrace();                                                             \
            throw std::runtime_error("Assertion " #expr " failed");                                \
        }                                                                                          \
    } while (0)

#define WARNCHK(expr)                                                                              \
    do {                                                                                           \
        if (!(expr)) {                                                                             \
            errchk_print_warning(__func__, __FILE__, __LINE__, #expr, "");                         \
        }                                                                                          \
    } while (0)

template <typename T>
void
errchk_print_debug(const std::string& label, const T& value)
{
    std::cout << label << ": " << value << std::endl;
}

template <typename T>
void
errchk_print_array(const std::string& label, const size_t count, const T* arr)
{
    std::cout << label << ": {";
    for (size_t i = 0; i < count; ++i)
        std::cout << arr[i] << (i + 1 < count ? ", " : "");
    std::cout << "}" << std::endl;
}

template <typename T>
std::ostream&
operator<<(std::ostream& os, const std::vector<T>& vec)
{
    os << "{\n";
    for (size_t i = 0; i < vec.size(); ++i)
        os << "\t" << i << ": " << vec[i] << (i + 1 < vec.size() ? ",\n" : "\n}");
    return os;
}

#define PRINT_DEBUG(value) (errchk_print_debug(#value, (value)))
#define PRINT_ARRAY_DEBUG(count, arr) (errchk_print_array(#arr, (count), (arr)))
