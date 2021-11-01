#pragma once
#include<cstdio>


// format of the read and write files:
/*
AcRealSize=<size in bytes>\n
Endianness=<LE or BE, no exotic stuff>\n
<binary dump of timesteps in order>
*/

// initialize a file for reading, NULL for failure. Will fail if size and endianness of program does not match the file.
FILE *open_read_file(const char *read_file_name);

FILE *internal_open_read_file(const char *read_file_name, int size, bool is_little_endian);

// initialize a file for writing. Overwrites an existing file if it exists
FILE *init_write_file(const char *write_file_name);

FILE *internal_init_write_file(const char *write_file_name, int size, bool is_little_endian);

// get the next timestep from a read-file opened with open_read_file. -1.0 for failure
template <class REAL>
int next_timestep_from_file(FILE *read_file, REAL *dt_out);


// append the timestep to a write-file initialized with open_write_file. -1 for failure
template <class REAL>
int record_timestep(FILE *write_file, REAL dt);

// for both reading and writing
int close_timestep_file(FILE *timestep_file);


// suppress and enable warnings and error messages 
//(for unit tests that cover error conditions)
//default is of course that warnings ENABLED
void enable_timestepfile_warnings();
void disable_timestepfile_warnings();