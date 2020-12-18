/*
#include "task.h"
#include <iostream>
#include <string>
#include <sstream>

void
logDiagnosticJSONMessage(std::string message_type, std::vector<std::pair<std::string,std::string>> members)
{
    std::stringstream outs;
    
    outs << "{"<< "\"message_type\":\""<<message_type<<"\"";
    for (auto& member : members){
        outs << ","<<"\""<<member.first<<"\":"<<member.second;
    }
    outs << "}" std::endl;
    std::cout << outs;
}

void
logStateTransitionEvent(std::string message_type, Task* t)
{
    std::vector<std::string> msg_members;
    msg_members.emplace_back("rank",t->rank);
    msg_members.emplace_back("substep",t->substep_counter.count);
}
*/

