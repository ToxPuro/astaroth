#pragma once

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ERROR(str)                                                                                 \
    {                                                                                              \
        fflush(stdout);                                                                            \
        time_t terr;                                                                               \
        time(&terr);                                                                               \
        fprintf(stderr,                                                                            \
                "\n\n\n\n┌──────────────────────── ERROR ───────────────────────────┐\n\n");       \
        fprintf(stderr, "%s", ctime(&terr));                                                       \
        fprintf(stderr, "Error in file '%s' line '%d', function '%s': %s\n", __FILE__, __LINE__,   \
                __func__, str);                                                                    \
        fprintf(stderr, "\n└──────────────────────── ERROR ───────────────────────────┘\n\n\n\n"); \
        fflush(stderr);                                                                            \
    }

#define WARNING(str)                                                                               \
    {                                                                                              \
        time_t terr;                                                                               \
        time(&terr);                                                                               \
        fprintf(stderr, "%s", ctime(&terr));                                                       \
        fprintf(stderr, "\tWarning in file '%s' line '%d', function '%s': %s\n", __FILE__,         \
                __LINE__, __func__, str);                                                          \
        fflush(stderr);                                                                            \
    }

// DO NOT REMOVE BRACKETS AROUND RETVAL. F.ex. if (!a < b) vs if (!(a < b)).
#define ERRCHK(retval)                                                                             \
    {                                                                                              \
        if ((retval) == false)                                                                     \
            ERROR(#retval " was false");                                                           \
    }

#define WARNCHK(retval)                                                                            \
    {                                                                                              \
        if ((retval) == false)                                                                     \
            WARNING(#retval " was false");                                                         \
    }
