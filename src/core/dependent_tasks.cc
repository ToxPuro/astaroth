#include "astaroth.h"

void Task::register_dependent(Task* t){
    dependents.push_back(t);
}

void Task::notify_dependents(int isubstep, AcReal dt){
    for (auto& d : dependents){
        d->notify(isubstep,dt);
    }
}

void Task::notify(int isubstep, AcReal dt){
    active_dependencies--;
    if(active_dependencies == 0 && allowed_triggers > 0){
        active_dependencies = total_dependencies;
        allowed_triggers--;
        execute(isubstep,dt);
    }
}

void Task::set_trigger_limit(size_t trigger_limit){
    allowed_triggers = trigger_limit;
}
