#include "simulation_soma_integration.h"
#include <cstdlib>
#include "astaroth.h"
#include <cassert>
#include <stdexcept>

#define OJL_FILE_IMPLEMENTATION
#include "ojl_file.hpp"

static size_t
comm_size(const MPI_Comm &c)
{
    int s;
    MPI_Comm_size(c, &i);
    return static_cast<size_t>(s);
}

soma_addressbook
discover_soma_addressbook()
{
    soma_addressbook ab{};
    {
        std::string soma_server_addr_file = slurp(std::getenv("SOMA_SERVER_ADDR_FILE"));
        size_t lstart = 0;
        size_t fend = soma_server_addr_file.size() - 1;
        while(size_t lend = soma_server_addr_file.find("\n", lstart) != std::string::npos && lstart < fend){
            
            if (lend == std::string::npos){
                lend = fend;
            }
            size_t space_loc = soma_server_addr_file.find(" ", lstart);
            if (space_loc == std::string::npos){
	        lstart = lend + 1;   
	        continue;
            }
            int rank = std::stoi(soma_server_addr_file.substr(lstart, space_loc));
            ab.addresses.push_back(soma_address{rank,soma_server_addr_file.substr(space_loc+1, lend)});
            lstart = lend + 1;   
        }
    }
    {
        std::string soma_node_addr_file = slurp(std::getenv("SOMA_NODE_ADDR_FILE"));
        size_t lstart = 0;
        size_t fend = soma_node_addr_file.size() - 1;
        while(size_t lend = soma_node_addr_file.find("\n", lstart) != std::string::npos && lstart < fend){
            if (lend == std::string::npos){
                lend = fend;
            }
            ab.addresses[rank] = soma_node_addr_file.substr(lstart, lend);
            lstart = lend + 1;   
        }
    }
    return ab;
}

size_t
get_soma_collector_idx(int pid)
{
    size_t num_servers = comm_size(MPI_COMM_WORLD) - comm_size(acGridMPIComm());

    int num_servers_per_instance;
    try {
        num_servers_per_instance = std::stoi(std::getenv("SOMA_NUM_SERVERS_PER_INSTANCE");
    } catch (std::exception e){
        // If we can't find a parameter,
	// assume all servers are part of a single instance
        num_servers_per_instance = num_servers;
    }

    int server_instance_id;
    try {
        server_instance_id = std::stoi(std::getenv("SOMA_SERVER_INSTANCE_ID");
    } catch (std::exception e){
        // If we can't find a parameter,
	// assume all servers are part of a single instance
        server_instance_id = 0;
    }
    assert(num_servers >= (server_instance_id+1)*num_servers_per_instance);
    assert(num_servers >= num_servers_per_instance);
    return server_instance_id*num_servers_per_instance + pid % num_servers_per_instance;
}

soma::CollectorHandle
discover_soma_collector(const std::string &protocol,
			const std::string &address,
		        const std::string &provider_id,
		        const std::string &collector)
{

    soma_addressbook = discover_soma_addressbook();
    
    soma::thallium::engine engine(protocol, THALLIUM_CLIENT_MODE);
    soma::Client client(engine);
    return client.makeCollectorHandle(address, provider_id,
		    	              soma::UUID::from_string(collector.c_str()));
}
