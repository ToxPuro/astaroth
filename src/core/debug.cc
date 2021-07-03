#if AC_MPI_ENABLED
#include "astaroth.h"
#include "astaroth_debug.h"
#include "task.h"
#include <stdio.h>

static void
graphWriteDependencies(FILE* fp, const AcTaskGraph* graph)
{
    // Compare that default_tasks == grid.default_tasks
    for (auto& task : graph->all_tasks) {
        for (auto& other : graph->all_tasks) {
            if (task->isPrerequisiteTo(other)) {
                // std::cout << other->name << " -> " << task->name << std::endl;
                fprintf(fp, "\t%s -> %s\n", other->name.c_str(), task->name.c_str());
            }
        }
    }
}

void
acGraphPrintDependencies(const AcTaskGraph* graph)
{
    graphWriteDependencies(stdout, graph);
}

void
acGraphWriteDependencies(const char* path, const AcTaskGraph* graph)
{
    FILE* fp;
    fp = fopen(path, "w");
    if (fp == NULL) {
        WARNING("Cannot open file to write graph dependencies.");
        fprintf(stderr, "file \"%s\" could not be opened\n", path);
        return;
    }

    graphWriteDependencies(fp, graph);
}

static void
graphWriteOrder(FILE* fp, const AcTaskGraph* graph)
{
    if ((*(graph->all_tasks.begin()))->rank == 0) {
        fprintf(fp, "Order\n");
        for (auto t : graph->all_tasks) {
            fprintf(fp, "\t%s\t%lu\n", t->name.c_str(), t->output_region->volume);
        }
    }
}

void
acGraphPrintOrder(const AcTaskGraph* graph)
{
    graphWriteOrder(stdout, graph);
}

void
acGraphWriteOrder(const char* path, const AcTaskGraph* graph)
{
    FILE* fp;
    fp = fopen(path, "w");
    if (fp == NULL) {
        WARNING("Cannot open file to write graph task order.");
        fprintf(stderr, "file \"%s\" could not be opened\n", path);
        return;
    }

    graphWriteOrder(fp, graph);
}

#define AC_DEBUG_TASK_TRANSITIONS (1)

#ifdef AC_DEBUG_TASK_TRANSITIONS
#include <iostream>
#endif
void
Task::logStateChangedEvent(std::string from, std::string to)
{
    #ifdef AC_DEBUG_TASK_TRANSITIONS
    if (rank == 0) {
    std::cout<< "{"
         <<"\"msg_type\":\"state_changed_event\","
         <<"\"rank\":"<<rank
         <<",\"substep\":"<<loop_cntr.i
         <<",\"order\":"<<order
         <<",\"task_type\":\""<<task_type<<"\""
         <<",\"tag\":"<<output_region->tag
         <<",\"seg_id\":["
             <<output_region->id.x<<","
             <<output_region->id.y<<","
             <<output_region->id.z<<"],"
         //<<"\"seg_type\":"<<output_region->facet_class<<","
         <<"\"from\":\""<<from<<"\""<<","
         <<"\"to\":\""<<to<<"\""
         <<"}"<<std::endl;
    }
    #endif
}
#endif
