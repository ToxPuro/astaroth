#include <cstdlib>
#include <cassert>
#include <stdexcept>

#include "astaroth.h"
#include "astaroth_utils.h"
#include "simulation_soma_integration.h"

#include <thallium.hpp>

#define OJL_FILE_IMPLEMENTATION
#include "ojl_file.hpp"

#define FMT_HEADER_ONLY
#include "fmt/format.h"

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
    acLogFromRootProc(pid, "Parsing SOMA address files\n");
    soma_addressbook lookup = discover_soma_addressbook();
    acLogFromRootProc(pid, "Done parsing SOMA address files\n");
    acLogFromRootProc(pid, "Getting server index\n");
    size_t server_idx       = get_soma_collector_idx(pid);
    acLogFromRootProc(pid, "Server index is: %lu\n", server_idx);
    std::string address     = lookup.addresses[server_idx].address;
    std::string collector   = lookup.collectors[server_idx];
    acLogFromRootProc(pid, "Address is: %s\n", address.c_str());
    acLogFromRootProc(pid, "Collector ID is: %s\n", collector.c_str());

    //Assume provider id is 0, since file does not specify it
    size_t provider_id = 0;

    acLogFromRootProc(pid, "Creating thallium engine\n");
    thallium::engine engine(protocol, THALLIUM_CLIENT_MODE);
    acLogFromRootProc(pid, "Creating soma client\n");
    soma::Client client(engine);
    acLogFromRootProc(pid, "Creating collector handle\n");
    return client.makeCollectorHandle(address, provider_id,
		    	              soma::UUID::from_string(collector.c_str()));
}


// Data collection
conduit::Node
query_local_diagnostics(const int pid, const AcMeshInfo info, const int timestep, const AcReal simulation_time)
{
    conduit::Node msg;
    msg["pid"]               = pid;
    msg["timestep"]          = timestep;
    msg["simulation_time"]   = simulation_time;

    Device device  = acGridGetDevice();

    //Synchronous call
    AcReal local_mass = 0;
    acDeviceReduceScalNotAveraged(device, STREAM_DEFAULT, RTYPE_EXP_SUM, Field(VTXBUF_LNRHO), &local_mass);

    AcReal dU = info.real_params[AC_dsx]*info.real_params[AC_dsy]*info.real_params[AC_dsz];
    msg["local_mass"] = local_mass*dU;

    //Local min reductions
    for (size_t i = 0; i < NUM_FIELDS;i++){
	std::string field_name = field_names[i];
	std::string val_key = fmt::format("{}/min/value", field_name);
	std::string loc_key = fmt::format("{}/min/location", field_name);

	AcMeshCell res = acDeviceMinElement(device, STREAM_DEFAULT, Field(i));

	msg[val_key] = res.value;
	msg[loc_key].set((int *)&res.location, 3);
	
	res = acDeviceMaxElement(device, STREAM_DEFAULT, Field(i));
        val_key = fmt::format("{}/max/value", field_name);
	loc_key = fmt::format("{}/max/location", field_name);

	msg[val_key] = res.value;
	msg[loc_key].set((int *)&res.location, 3);

	AcMeshBooleanSearchResult nan_res = acDeviceFirstNANElement(device, STREAM_DEFAULT, Field(i));
        val_key = fmt::format("{}/nan/value", field_name);
        loc_key = fmt::format("{}/nan/location", field_name);
	msg[val_key] = nan_res.value;
	msg[loc_key].set((int *)&nan_res.location, 3);
    }
    return msg;
}


