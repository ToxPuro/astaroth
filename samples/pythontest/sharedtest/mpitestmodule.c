#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <mpi.h>

static float myglobal[10];

static PyObject *
mpitest_init(PyObject *self, PyObject *args) {
    float initnum;
    int ok;
    initnum = 0; 
    ok = PyArg_ParseTuple(args, "f", initnum);
    
    //MPI_Init(NULL, NULL);
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    myglobal[0] = initnum;

    printf("nprocs = %i, pid = %i, myglobal[0] = %f \n", nprocs, pid, myglobal[0]);

    //MPI_Finalize();

    return Py_BuildValue("i", ok);
}

/*
static PyObject *
mpitest_multiply(PyObject *self, PyObject *args) {
    int ok; double coeff;
    ok = PyArg_ParseTuple(args, "d", &coeff);
    myglobal.n = myglobal.n*(int) coeff; 
    myglobal.b = myglobal.b*(float) coeff;
    myglobal.f = myglobal.f*(double) coeff; 

    return Py_BuildValue("i", ok);
}

static PyObject *
mpitest_print(PyObject *self, PyObject *args) {
    printf("%i, %f, %e\n", myglobal.n, myglobal.b, myglobal.f);
    return Py_BuildValue("i", 1);
}

*/

static PyMethodDef JusttestMethods[] = {
    {"init",  mpitest_init, METH_VARARGS,
     "Initialize global truct"},
    //{"multiply",  mpitest_multiply, METH_VARARGS,
    // "Multiply global truct"},
    //{"print",  mpitest_print, METH_VARARGS,
    // "print global struct"},
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
