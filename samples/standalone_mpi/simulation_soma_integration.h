#ifndef SIMULATION_SOMA_INTEGRATION_H
#define SIMULATION_SOMA_INTEGRATION_H

#include <soma/Client.hpp>
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

soma_addressbook discover_soma_address();

soma::CollectorHandle
discover_soma_collector(const std::string &protocol,
			const std::string &address,
		        const std::string &provider_id,
		        const std::string &collector);

#endif
