#ifndef OJL_FILE_HPP
#define OJL_FILE_HPP

/* Copyright Oskar Lappi 2023
 *
 * Usage of the works is permitted provided that this instrument is retained
 * with the works, so that any entity that uses the works is notified of this
 * instrument.
 *
 * DISCLAIMER: THE WORKS ARE WITHOUT WARRANTY.
 */

// FILE, fwrite, fopen
#include <cstdio>

// POSIX fstat
#include <sys/stat.h>

// POSIX dirname
#include <libgen.h>

// C++ filesystem API
#include <filesystem>

// For a string interface
#include <string>
// For a uint8_t vec interface
#include <vector>

// unique_ptr
#include <memory>

// File resource management
enum class FileState { Closed = 0, Open, Error };

struct file_closer {
    void operator()(std::FILE *fp);
};

using unique_file_ptr = std::unique_ptr<std::FILE, file_closer>;

struct file_handle {
    unique_file_ptr m_fileptr;
    std::string     m_filename;
    FileState       m_filestate;

    void open(const std::string &filename_, const std::string &mode);
    void close();

    void        write(const std::string &text);
    std::vector<uint8_t> bslurp() const;
    std::string slurp() const;
    void        flush() const;
};

//Convenience function for reading a file
std::string slurp(std::string filepath);
std::vector<uint8_t> bslurp(std::string filepath);

#ifdef OJL_FILE_IMPLEMENTATION
void
file_closer::operator()(std::FILE *fp)
{
    if (fp == nullptr){
        return;
    }
    int ret = std::fclose(fp); //NOLINT(cppcoreguidelines-owning-memory)
    if (ret != 0){
        // TODO: throw?
        fprintf(stderr, "Error closing file (file_closer)\n");
    }
}

void
file_handle::open(const std::string &filename, const std::string &mode)
{
        close();
	m_filename = filename;

	// TODO!!!: should only be done when writing to a file
	std::filesystem::path p = std::filesystem::u8path(m_filename);
        if (p.has_parent_path()){
	    std::filesystem::create_directories(p.parent_path());
	}

	m_fileptr = unique_file_ptr(fopen(m_filename.c_str(), mode.c_str()));
	if (!m_fileptr) {
	    // TODO: throw?
	    fprintf(stderr, "Error opening file %s\n", m_filename.c_str());
	    m_filestate = FileState::Error;
	    return;
	}
	m_filestate = FileState::Open;

}

void
file_handle::close()
{
	m_fileptr.reset();
	m_filestate = FileState::Closed;
}

void
file_handle::write(const std::string &text)
{
	if (fwrite(text.data(), 1, text.length(), m_fileptr.get()) != text.length()) {
	    fprintf(stderr, "Error writing to file %s\n", m_filename.c_str());
            m_filestate = FileState::Error;
	}
}

std::vector<uint8_t>
file_handle::bslurp() const
{
	if (!m_fileptr){
	    fprintf(stderr, "Error: trying to read a null filehandle %s\n", m_filename.c_str());
            return {};
        }

	int         fd = fileno(m_fileptr.get());
        struct stat statbuf {
        };
        fstat(fd, &statbuf);

        size_t file_len = statbuf.st_size;
        std::vector<uint8_t> data(file_len, 0);

        if (data.size() != file_len) {
            fprintf(stderr, "Error allocating memory while reading file %s\n", m_filename.c_str());
            return data;
        }

        size_t n = 0;
        if ((n = fread(data.data(), file_len, 1, m_fileptr.get())) != 1) {
            // TODO: throw?, error reading
	    //m_filestate = FileState::Error;
            fprintf(stderr, "Error reading file, n = %lu\n", n);
        }
        return data;
}

std::string
file_handle::slurp() const
{
        std::string str;
	if (!m_fileptr){
	    fprintf(stderr, "Error: trying to read a null filehandle %s\n", m_filename.c_str());
            return str;
        }

	int         fd = fileno(m_fileptr.get());
        struct stat statbuf {
        };
        fstat(fd, &statbuf);

        size_t file_len = statbuf.st_size;
        char  *filebuf  = new char[file_len + 1]; // NOLINT(cppcoreguidelines-owning-memory)

        if (filebuf == nullptr) {
            fprintf(stderr, "Error allocating memory while reading file %s\n", m_filename.c_str());
            // TODO: throw?
            delete[] filebuf; // NOLINT(cppcoreguidelines-owning-memory)
            return str;
        }

        size_t n = 0;
        if ((n = fread(filebuf, file_len, 1, m_fileptr.get())) != 1) {
            // TODO: throw?, error reading
	    //m_filestate = FileState::Error;
            fprintf(stderr, "Error reading file, n = %lu\n", n);
        }
        filebuf[file_len] = '\0'; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        str               = filebuf;
        delete[] filebuf; // NOLINT(cppcoreguidelines-owning-memory)
        return str;
}

void
file_handle::flush() const
{
        if (fflush(m_fileptr.get()) != 0) {
	    fprintf(stderr, "Error: can't flush file %s\n",m_filename.c_str());
            // TODO: throw
        }
}

std::string
slurp(std::string filepath)
{
    file_handle f{};
    f.open(filepath, "r");
    return f.slurp();
}

std::vector<uint8_t>
bslurp(std::string filepath)
{
    file_handle f{};
    f.open(filepath, "rb");
    return f.bslurp();
}
#endif
#endif
