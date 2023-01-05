#include <cstdint>
#include <cstdio>
#include <ctime>
#include <cstdarg>

#include <sys/stat.h>
#include <unistd.h>

#include <string>

//Structure for keeping track of any generic condition
struct SimulationPeriod {
    int    step_period;
    AcReal time_period;
    AcReal time_threshold;

    SimulationPeriod()
    : step_period(0), time_period(0), time_threshold(0)
    {}


    SimulationPeriod(int s, AcReal p)
    : step_period(s), time_period(p), time_threshold(p)
    {}

    SimulationPeriod(int s, AcReal p, AcReal threshold_offset)
    : step_period(s), time_period(p), time_threshold(p + threshold_offset)
    {}
    bool
    check(int time_step, AcReal time)
    {
       if ((time_period > 0 && time >= time_threshold) ||
          (step_period > 0 && time_step % step_period == 0)){
           time_threshold += time_period;
           return true;
       }

        return false;
    }
};


//std::string to_timestamp(time_t t) {
//  char timestamp[80];
//  strftime(timestamp, 80, "%Y-%m-%d-%I:%M:%S", localtime(&t));
//  return std::string(timestamp);
//}

// Structure for keeping track of a file used by the user to signal things
struct UserSignalFile {
  std::string file_path;
  time_t mod_time;

  UserSignalFile()
   : file_path{""}, mod_time{0}
  {}


  UserSignalFile(std::string filename)
      : file_path(filename), mod_time{stat_file_mod_time()} {};

  bool file_exists() { return access(file_path.c_str(), F_OK) == 0; }

  time_t stat_file_mod_time() {
    struct stat s;
    if (stat(file_path.c_str(), &s) == 0) {
      return s.st_mtime;
    }
    return 0;
  }

  bool check() {
    time_t statted_mod_time = stat_file_mod_time();
    time_t prev_mod_time    = mod_time;
    mod_time                = std::max(statted_mod_time, prev_mod_time);
    return statted_mod_time > prev_mod_time;
  };
};


// Logging in specific formats
constexpr size_t sim_log_msg_len = 512;
static size_t sim_tstamp_len = 0;
static char sim_log_msg[sim_log_msg_len] = "";

void
set_simulation_timestamp(int step, AcReal time)
{
    //TODO: only set step and time, and lazily create the log stamp whenever it's needed
    snprintf(sim_log_msg, sim_log_msg_len, "[i:%d, t:%.2e] ", step, time);
    sim_tstamp_len = strlen(sim_log_msg);
}

void
log_from_root_proc_with_sim_progress(int pid, std::string msg, ...)
{
    if (pid == 0){
      strncpy(sim_log_msg+sim_tstamp_len, msg.c_str(), sim_log_msg_len - sim_tstamp_len);
      va_list args;
      va_start(args, msg);
      acVA_LogFromRootProc(pid, sim_log_msg, args); 
      va_end(args);
    }
}

void
debug_log_from_root_proc_with_sim_progress(int pid, std::string msg, ...)
{
#ifndef NDEBUG
   if (pid == 0){
      strncpy(sim_log_msg+(sim_tstamp_len), msg.c_str(), sim_log_msg_len - sim_tstamp_len);
      va_list args;
      va_start(args, msg);
      acVA_DebugFromRootProc(pid, sim_log_msg, args);
      va_end(args);
   }
#else
   (void)pid;
   (void)msg;
#endif
}
