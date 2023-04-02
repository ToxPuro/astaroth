#include <cstdlib>
#include <cassert>
#include <stdexcept>

#include "astaroth.h"
#include "simulation_soma_integration.h"

#include <thallium.hpp>

#define OJL_FILE_IMPLEMENTATION
#include "ojl_file.hpp"

std::vector<soma_address>
parse_address_file(std::string fpath)
{
    std::vector<soma_address> addresses;

    std::string soma_server_addr_file = slurp(fpath);

    size_t lstart = 0;
    size_t fend = soma_server_addr_file.size() - 1;
    while(true){
        size_t lend = soma_server_addr_file.find("\n", lstart);
        if (lend == std::string::npos){
                lend = fend;
        }
        size_t space_loc = soma_server_addr_file.find(" ", lstart);
        if (space_loc == std::string::npos || space_loc > lend){
            lstart = lend + 1;   
            continue;
        }
        int rank         = std::stoi(soma_server_addr_file.substr(lstart, space_loc - lstart));
        std::string addr = soma_server_addr_file.substr(space_loc+1, lend - space_loc -1);
        addresses.push_back(soma_address{rank,addr});
        lstart = lend + 1;
        if (lstart >= fend){
            break;
        }
    }
    return addresses;
}

std::vector<std::string>
parse_collector_file(std::string fpath)
{
    std::vector<std::string> collectors;

    std::string soma_node_addr_file = slurp(fpath);

    size_t lstart = 0;
    size_t fend = soma_node_addr_file.size() - 1;
    while(true){
        size_t lend = soma_node_addr_file.find("\n", lstart);
        if (lend == std::string::npos){
            lend = fend;
        }
        collectors.push_back(soma_node_addr_file.substr(lstart, lend - lstart));
        lstart = lend + 1;   
        if (lstart >= fend){
            break;
        }
    }
    return collectors;
}

soma_addressbook
discover_soma_addressbook()
{
    //Files assumed to be in a valid state at this point
    //Check with SOMA that the files have been written, e.g. with a Barrier
    printf("Discovering addressbook\n");
    soma_addressbook ab{};
    {
        std::string server_addr_fpath = std::getenv("SOMA_SERVER_ADDR_FILE");
        ab.addresses = parse_address_file(server_addr_fpath);
    }

    printf("Discovering collector id's\n");
    {
        std::string node_addr_fpath = std::getenv("SOMA_NODE_ADDR_FILE");
        ab.collectors = parse_collector_file(node_addr_fpath);
    }
    // assert(ab.addresses.size() == ab.collectors.size());
    return ab;
}


void
print_soma_addressbook(const soma_addressbook &lookup)
{
    printf("SOMA addresses\n");
    for (const auto &addr: lookup.addresses){
        printf("  %d %s\n", addr.rank, addr.address.c_str());
    }
    printf("SOMA collector IDs\n");
    for (const auto &coll: lookup.collectors){
        printf("  %s\n", coll.c_str());
    }
}



static size_t
comm_size(const MPI_Comm &c)
{
    int s;
    MPI_Comm_size(c, &s);
    return static_cast<size_t>(s);
}



size_t
get_soma_collector_idx(int pid)
{
    size_t num_servers = comm_size(MPI_COMM_WORLD) - comm_size(acGridMPIComm());

    int num_servers_per_instance;
    try {
        num_servers_per_instance = std::stoi(std::getenv("SOMA_NUM_SERVERS_PER_INSTANCE"));
    } catch (std::exception e){
        // If we can't find a parameter,
	// assume all servers are part of a single instance
        num_servers_per_instance = num_servers;
    }
    assert(num_servers_per_instance > 0);

    int server_instance_id;
    try {
        server_instance_id = std::stoi(std::getenv("SOMA_SERVER_INSTANCE_ID"));
    } catch (std::exception e){
        // If we can't find a parameter,
	// assume all servers are part of a single instance
        server_instance_id = 0;
    }
    assert(num_servers == 0 || num_servers >= (server_instance_id+1)*num_servers_per_instance);
    assert(num_servers == 0 || num_servers >= num_servers_per_instance);
    return server_instance_id*num_servers_per_instance + pid % num_servers_per_instance;
}

void
log_soma_config(const soma_addressbook &lookup, int pid)
{
    auto ac_comm = acGridMPIComm();
    size_t n_astaroth_ranks = comm_size(ac_comm);

    if (pid == 0){
        printf("SOMA config\n");
        print_soma_addressbook(lookup);
    }
    size_t server_idx = get_soma_collector_idx(pid);
    for (size_t i = 0; i < n_astaroth_ranks; i++){
	if (pid == i){
            printf("Astaroth rank %d\n"
		   "------------------\n"
		   "   SOMA rank: %d\n"
		   "     address: %s\n"
		   "   collector: %s\n"
		   "\n",
		   pid,
		   lookup.addresses[server_idx].rank,
		   lookup.addresses[server_idx].address.c_str(),
		   lookup.collectors[server_idx].c_str()
		   );
	}
	MPI_Barrier(ac_comm);
    }
}

void
log_soma_config(int pid)
{
    log_soma_config(discover_soma_addressbook(), pid);
}


soma::CollectorHandle
discover_soma_collector(const std::string &protocol,
			const int pid)
{

    // Sync with SOMA server
    MPI_Barrier(MPI_COMM_WORLD);
    soma_addressbook lookup = discover_soma_addressbook();
    size_t server_idx       = get_soma_collector_idx(pid);
    std::string address = lookup.addresses[server_idx].address;
    std::string collector   = lookup.collectors[server_idx];

    //Assume provider id is 0, since file does not specify it
    size_t provider_id = 0;

    thallium::engine engine(protocol, THALLIUM_CLIENT_MODE);
    soma::Client client(engine);
    return client.makeCollectorHandle(address, provider_id,
		    	              soma::UUID::from_string(collector.c_str()));
}
