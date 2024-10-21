#pragma once
#include "errchk.h"

#define ERRCHK(expr)                                                                               \
    do {                                                                                           \
        if (!(expr)) {                                                                             \
            errchk_print_error(__func__, __FILE__, __LINE__, #expr, "");                           \
            throw std::runtime_error("General error");                                             \
        }                                                                                          \
    } while (0)

#define HANDLE(expr)                                                                               \
    try {                                                                                          \
        expr;                                                                                      \
    }                                                                                              \
    catch (const std::exception& e) {                                                              \
        ERROR_DESC("Exception %s caught", e.what());                                               \
        throw;                                                                                     \
    }
