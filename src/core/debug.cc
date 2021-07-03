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

#define AC_DEBUG_TASK_TRANSITIONS (0)

void
Task::logStateChangedEvent(const char* from, const char* to)
{
    //This line is here to stop the compiler from complaining about unused variables
    if (from == to){;}

    #if AC_DEBUG_TASK_TRANSITIONS == 1
    printf("{"
           "\"msg_type\":\"state_changed_event\","
           "\"rank\":%d,"
           "\"iteration\":%lu,"
           "\"task_order\":%d,"
           "\"task_type\":%d,"
           "\"tag\":%d,"
           "\"region_id\":[%d,%d,%d],"
           "\"from\":\"%s\","
           "\"to\":\"%s\""
           "}\n",
           rank, loop_cntr.i, order, task_type, output_region->tag,
           output_region->id.x, output_region->id.y, output_region->id.z,
           from, to);
    #endif
}
#endif
