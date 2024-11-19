#pragma once
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

#define ERRCHK_EXPR_DESC(expr, ...)                                                                \
    do {                                                                                           \
        if (!(expr)) {                                                                             \
            errchk_print_error(__func__, __FILE__, __LINE__, #expr, "");                           \
            errchk_print_stacktrace();                                                             \
            throw std::runtime_error("Assertion " #expr " failed");                                \
        }                                                                                          \
    } while (0)

#include <chrono>
#define BENCHMARK(cmd)                                                                             \
    do {                                                                                           \
        const auto start__{std::chrono::system_clock::now()};                                     \
        (cmd);                                                                                     \
        const auto ms_elapsed__ = std::chrono::duration_cast<std::chrono::milliseconds>(           \
            std::chrono::system_clock::now() - start__);                                           \
        std::cout << "[" << ms_elapsed__.count() << " ms] " << #cmd << std::endl;                  \
    } while (0)
