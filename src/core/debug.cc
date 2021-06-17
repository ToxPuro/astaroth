#if AC_MPI_ENABLED
#include "astaroth.h"
#include "astaroth_debug.h"
#include "task.h"
#include <stdio.h>

static void
graphWriteDependencies(FILE* fp, const TaskGraph* graph)
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
acGraphPrintDependencies(const TaskGraph* graph)
{
    graphWriteDependencies(stdout, graph);
}

void
acGraphWriteDependencies(const char* path, const TaskGraph* graph)
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
graphWriteOrder(FILE* fp, const TaskGraph* graph)
{
    if ((*(graph->all_tasks.begin()))->rank == 0) {
        fprintf(fp, "Order\n");
        for (auto t : graph->all_tasks) {
            fprintf(fp, "\t%s\t%lu\n", t->name.c_str(), t->output_region->volume);
        }
    }
}

void
acGraphPrintOrder(const TaskGraph* graph)
{
    graphWriteOrder(stdout, graph);
}

void
acGraphWriteOrder(const char* path, const TaskGraph* graph)
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

/*
void
Task::logStateChangedEvent(std::string from, std::string to)
{
    //NOTE: the keys used here don't reflect terminology in Astaroth
    //because the messages are read by a python tool which expects these keys.
    std::cout<< "{"
         <<"\"msg_type\":\"state_changed_event\","
         <<"\"rank\":"<<rank
         <<",\"substep\":"<<loop_cntr.i
         <<",\"task_type\":\""<<task_type<<"\""
         <<",\"tag\":"<<output_region->tag
         <<",\"seg_id\":["
             <<output_region->id.x<<","
             <<output_region->id.y<<","
             <<output_region->id.z<<"],"
         <<"\"seg_type\":"<<output_region->facet_class<<","
         <<"\"from\":\""<<from<<"\""<<","
         <<"\"to\":\""<<to<<"\""
         <<"}"<<std::endl;
}
*/
#endif
