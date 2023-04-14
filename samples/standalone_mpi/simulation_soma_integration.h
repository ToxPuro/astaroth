#ifndef SIMULATION_SOMA_INTEGRATION_H
#define SIMULATION_SOMA_INTEGRATION_H

#include <soma/Client.hpp>
#include <conduit/conduit.hpp>
#include <string>
#include <map>

struct soma_address {
    int rank;
    std::string address;
};

//Since SOMA spits out two files, we probably want to read them like this, in case they are broken
struct soma_addressbook {
   std::vector<soma_address> addresses;  //Thallium engine addresses
   std::vector<std::string> collectors; //Collector UUIDs
};

std::vector<soma_address> parse_address_file(std::string fpath);
std::vector<std::string>  parse_collector_file(std::string fpath);
std::string               parse_protocol(const std::string &address);

soma_addressbook discover_soma_addressbook();
size_t get_soma_collector_idx(int pid);

void print_soma_addressbook(const soma_addressbook &lookup);
void log_soma_config(const soma_addressbook &lookup, int pid);
void log_soma_config(int pid);

soma::CollectorHandle discover_soma_collector(int pid);

// Data publishing
conduit::Node query_local_diagnostics(const int pid, const AcMeshInfo info, const int timestep, const AcReal simulation_time);

#endif
