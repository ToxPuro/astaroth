#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <mpi.h>

#define MYARR 16

#define DEBUG 1 

static float myglobal[MYARR];

static PyObject *
mpitest_mpiinit(PyObject *self, PyObject *args) {

    MPI_Init(NULL,NULL);

    return Py_BuildValue("i", 1);
}

static PyObject *
mpitest_mpifinalize(PyObject *self, PyObject *args) {

    MPI_Finalize();

    return Py_BuildValue("i", 1);
}


static PyObject *
mpitest_setup(PyObject *self, PyObject *args) {
    float initnum;
    int ok;
    initnum = 0; 
    ok = PyArg_ParseTuple(args, "f", initnum);
    
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    initnum = pid*MYARR; 

    myglobal[0] = initnum;

    printf("Initializing nprocs = %i, pid = %i, myglobal[0] = %f \n", nprocs, pid, myglobal[0]);

    return Py_BuildValue("i", ok);
}

static PyObject *
mpitest_makeseries(PyObject *self, PyObject *args) {
    int ok, ind, pid; double coeff;
    ok = PyArg_ParseTuple(args, "d", &coeff);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    for (ind = 0; ind < MYARR; ind++) {
        myglobal[ind] =  ((float) pid * (float) MYARR ) + ((float) ind*(float) coeff);
    } 

    return Py_BuildValue("i", ok);
}

static PyObject *
mpitest_print(PyObject *self, PyObject *args) {
    int ind, pid; 
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    printf("Array, pid = %i, [ ", pid) ;
    for (ind = 0; ind < MYARR; ind++) {
        printf("%.1f", myglobal[ind]) ;
        if (ind == MYARR-1) {
            printf(" ] \n") ;
        } else {
            printf(", ");
        }
    }
    return Py_BuildValue("i", 1);
}

static PyObject *
mpitest_barrier(PyObject *self, PyObject *args) {

    MPI_Barrier(MPI_COMM_WORLD);

    return Py_BuildValue("i", 1);
}

static PyObject *
mpitest_copyval(PyObject *self, PyObject *args) {
    int ok, element;
    ok = PyArg_ParseTuple(args, "i", &element);

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    float number, sendn;
    int send_ind, recv_ind;
    send_ind = pid+1;
    recv_ind = pid-1;
    if (send_ind >= nprocs) {send_ind = send_ind - nprocs;}
    if (recv_ind < 0) {recv_ind = recv_ind + nprocs;}

#if DEBUG
    printf("pid %i, send_ind %i, recv_ind %i \n", pid, send_ind, recv_ind);
#endif

    sendn = myglobal[element];

    MPI_Send(&sendn,  1, MPI_FLOAT, send_ind, 0, MPI_COMM_WORLD);
    MPI_Recv(&number, 1, MPI_FLOAT, recv_ind, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    myglobal[MYARR-1-element] = number;

    return Py_BuildValue("i", ok);
}


static PyMethodDef JusttestMethods[] = {
    {"setup",  mpitest_setup, METH_VARARGS,
     "Initialize global truct"},
    {"makeseries",  mpitest_makeseries, METH_VARARGS,
     "Set values on array"},
    {"print",  mpitest_print, METH_VARARGS,
     "print global struct"},
    {"barrier",  mpitest_barrier, METH_VARARGS,
     "barrier"},
    {"copyval",  mpitest_copyval, METH_VARARGS,
     "copyval"},
    {"mpiinit",  mpitest_mpiinit, METH_VARARGS,
     "mpiinit"},
    {"mpifinalize",  mpitest_mpifinalize, METH_VARARGS,
     "mpifinalize"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef mpitestmodule = {
    PyModuleDef_HEAD_INIT,
    "mpitest",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    JusttestMethods
};


PyMODINIT_FUNC
PyInit_mpitest(void)
{
    return PyModule_Create(&mpitestmodule);
}

int
main(int argc, char *argv[])
{
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    /* Add a built-in module, before Py_Initialize */
    if (PyImport_AppendInittab("mpitest", PyInit_mpitest) == -1) {
        fprintf(stderr, "Error: could not extend in-built modules table\n");
        exit(1);
    }

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(program);

    /* Initialize the Python interpreter.  Required.
       If this step fails, it will be a fatal error. */
    Py_Initialize();

    /* Optionally import the module; alternatively,
       import can be deferred until the embedded script
       imports it. */
    PyObject* pmodule = PyImport_ImportModule("mpitest");
    if (!pmodule) {
        PyErr_Print();
        fprintf(stderr, "Error: could not import module 'mpitest'\n");
    }

    PyMem_RawFree(program);
    return 0;
}
