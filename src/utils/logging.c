#include "astaroth_utils.h"
#include <stdarg.h>
#include <string.h>
#include <time.h>

//Logging utils
void
acLogFromRootProc(const int pid, const char* msg, ...)
{
    if (pid == 0) {
        time_t now  = time(NULL);
        char* timestamp  = ctime(&now);
        size_t stamp_len = strlen(timestamp);
        // Remove trailing newline
        timestamp[stamp_len - 1] = '\0';
        // We know the exact length of the timestamp (26 chars), so we could force this function to
        // take chars with a 26 prefix blank buffer
        fprintf(stderr, "%s : ", timestamp);

        va_list args;
        va_start(args, msg);
        vfprintf(stderr, msg, args);
        fflush(stderr);
        va_end(args);
    }
}

void
acVerboseLogFromRootProc(const int pid, const char* msg, ...)
{
#if AC_VERBOSE
    if (pid == 0) {
        time_t now  = time(NULL);
        char* timestamp  = ctime(&now);
        size_t stamp_len = strlen(timestamp);
        // Remove trailing newline
        timestamp[stamp_len - 1] = '\0';
        // We know the exact length of the timestamp (26 chars), so we could force this function to
        // take chars with a 26 prefix blank buffer
        fprintf(stderr, "%s : ", timestamp);

        va_list args;
        va_start(args, msg);
        vfprintf(stderr, msg, args);
        fflush(stderr);
        va_end(args);
    }
#endif
}

void acDebugFromRootProc(const int pid, const char* msg, ...)
{
#ifndef NDEBUG
    if (pid == 0) {
        time_t now  = time(NULL);
        char* timestamp  = ctime(&now);
        size_t stamp_len = strlen(timestamp);
        // Remove trailing newline
        timestamp[stamp_len - 1] = '\0';
        // We know the exact length of the timestamp (26 chars), so we could force this function to
        // take chars with a 26 prefix blank buffer
        fprintf(stderr, "%s : ", timestamp);

        va_list args;
        va_start(args, msg);
        vfprintf(stderr, msg, args);
        fflush(stderr);
        va_end(args);
    }
#endif

}

