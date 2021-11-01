#include "timestepfile.h"
#include <stdint.h>
#include <assert.h>
#include <cstring>
#include <climits>
#include "astaroth.h"

static bool show_warnings = true;

void enable_timestepfile_warnings() {
    show_warnings = true;
}

void disable_timestepfile_warnings() {
    show_warnings = false;
}

#define ERR_PRINT(...) {if (show_warnings) { fprintf(stderr, __VA_ARGS__); }}

bool is_le() {

    // this only works if a char is 8 bit but who cares
    assert(CHAR_BIT==8);
    uint16_t a = 0x0010;
    char *b = (char *)&a;
    return (b[0]==0x10);
}

// initialize a file for reading, NULL for failure. Will fail if size and endianness of program does not match the file.
FILE *open_read_file(const char *read_file_name){

    printf("sizeof real when calling internal read: %ld\n", sizeof(AcReal));
    return internal_open_read_file(read_file_name, sizeof(AcReal), is_le());
}

// initialize a file for writing
FILE *init_write_file(const char *write_file_name){
    return internal_init_write_file(write_file_name, sizeof(AcReal), is_le());
}

FILE *internal_open_read_file(const char *read_file_name, int size, bool is_little_endian){

    printf("in internal_open reader: sizeof real is %d\n", size);

    // check if file matches expectations
    FILE *f = fopen(read_file_name, "r+b");
    int size_onfile;
    char endian_desc[3];
    int succ = fscanf(f, "AcRealSize=%d;", &size_onfile);
    if (succ != 1) {
        ERR_PRINT("file format of %s does not seem to match expectations, the first line should be \"AcRealSize=<size in bytes as a decimal>;\"", read_file_name);
        exit(1);
    }
    printf("size on file after fscanf is %d\n", size_onfile);
    fscanf(f, "Endianness=%2s;", endian_desc);
    printf("size on file after fscanf is %d\n", size_onfile);
    if (succ != 1) {
        ERR_PRINT("file format of %s does not seem to match expectations, the second line should be \"Endianness=<LE or BE>;\"\n", read_file_name);
        exit(1);
    }
    printf("size=%d; size_onfile=%d\n", size, size_onfile);
    if (size != size_onfile) {
        ERR_PRINT("file %s holds %d-byte floats, but this computer uses %d-byte floats\n", read_file_name, size_onfile, size);
        exit(1);
    }

    bool is_be = !strcmp(endian_desc, "BE");
    bool is_le = !strcmp(endian_desc, "LE");
    if (!is_be && !is_le) {
        ERR_PRINT("file %s could not be identified as being little or big endian (endianness in file is \"%s\"), make sure the second line is \"Endianness=<LE or BE>\"\n", read_file_name, endian_desc);
        return NULL;
    }
    if (is_le != is_little_endian) {
        ERR_PRINT("endianness in file %s is %s, but this does not match the endianness of the current system\n", read_file_name, endian_desc);
        return NULL;
    }


    // actually open file and move to beginning of binary dump
    //f = fopen(read_file_name, "r+b");

    return f;
}



FILE *internal_init_write_file(const char *write_file_name, int size, bool is_little_endian){
    FILE *f = fopen(write_file_name, "w");
    // write some info

    fprintf(f, "AcRealSize=%d;", size);
    const char * endian_str = is_little_endian ? "LE" : "BE";
    fprintf(f, "Endianness=%s;", endian_str);

    fclose(f);
    f = fopen(write_file_name, "a+b");
    return f;
}

// get the next timestep from a read-file opened with open_read_file. -1.0 for failure
template <class REAL>
int next_timestep_from_file(FILE *read_file, REAL *dt_out){

    char *mem;
    mem = (char*) dt_out;
    
    for (unsigned i=0; i<sizeof(REAL); i++){
        int read = fscanf(read_file, "%c", &mem[i]);
        if (read != 1) {
            fprintf(stderr, "fscanf from timestepfile failed on i=%d with return value %d, aborting\n", i, read);
            abort();
        }
    }

    printf("\nread timestep from file: ");
    for (unsigned i=0; i<sizeof(REAL); i++){
        printf("(%X)", mem[i]);
    }
    printf(" |||---> %lf\n", double(*dt_out));


    return 0;
}

// append the timestep to a write-file initialized with open_write_file. -1 for failure
template <class REAL>
int record_timestep(FILE *write_file, REAL dt){
    char *mem;
    mem = (char*)&dt;
    for (unsigned i=0; i<sizeof(REAL); i++){
        fprintf(write_file, "%c", mem[i]);
    }
    return 0;
}

// for both reading and writing
int close_timestep_file(FILE *timestep_file){
    return fclose(timestep_file);
}

template int record_timestep<double>(FILE*,double);
template int record_timestep<float>(FILE*,float);

template int next_timestep_from_file<double>(FILE*,double*);
template int next_timestep_from_file<float>(FILE*,float*);