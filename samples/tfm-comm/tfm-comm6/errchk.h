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
#define PRINT_DEBUG(value) (errchk_print_debug(#value, (value)))
