#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ERROR(str)                                                                                 \
    {                                                                                              \
        time_t terr;                                                                               \
        time(&terr);                                                                               \
        fprintf(stderr,                                                                            \
                "\n\n\n\n┌──────────────────────── ERROR ───────────────────────────┐\n\n");       \
        fprintf(stderr, "%s", ctime(&terr));                                                       \
        fprintf(stderr, "Error in file %s line %d: %s\n", __FILE__, __LINE__, str);                \
        fprintf(stderr, "\n└──────────────────────── ERROR ───────────────────────────┘\n\n\n\n"); \
        fflush(stderr);                                                                            \
    }

#define WARNING(str)                                                                               \
    {                                                                                              \
        time_t terr;                                                                               \
        time(&terr);                                                                               \
        fprintf(stderr, "%s", ctime(&terr));                                                       \
        fprintf(stderr, "\tWarning in file %s line %d: %s\n", __FILE__, __LINE__, str);            \
        fflush(stderr);                                                                            \
    }

// DO NOT REMOVE BRACKETS AROUND RETVAL. F.ex. if (!a < b) vs if (!(a < b)).
#define ERRCHK(retval)                                                                             \
    {                                                                                              \
        if (!(retval))                                                                             \
            ERROR(#retval " was false");                                                           \
    }

#define WARNCHK(retval)                                                                            \
    {                                                                                              \
        if (!(retval))                                                                             \
            WARNING(#retval " was false");                                                         \
    }
