/* Log a message with a timestamp from the root proc (if pid == 0) */
void acLogFromRootProc(const int pid, const char* msg, ...);
void acVA_LogFromRootProc(const int pid, const char* msg, va_list args);

/* Log a message with a timestamp from the root proc (if pid == 0) if the build flag VERBOSE is on
 */
void acVerboseLogFromRootProc(const int pid, const char* msg, ...);
void acVA_VerboseLogFromRootProc(const int pid, const char* msg, va_list args);

/* Log a message with a timestamp from the root proc (if pid == 0) in a debug build */
void acDebugFromRootProc(const int pid, const char* msg, ...);
void acVA_DebugFromRootProc(const int pid, const char* msg, va_list arg);
