#define PY_SSIZE_T_CLEAN
#include <Python.h>

//Besed on https://docs.python.org/3.8/extending/extending.html

typedef struct {
    int n;
    float b;
    double f;
} myStruct;


static myStruct myglobal = {};

static PyObject *
justtest_init(PyObject *self, PyObject *args) {
    int n, ok; float b; double f;
    ok = PyArg_ParseTuple(args, "ifd", &n, &b, &f);
    myglobal.n = n; 
    myglobal.b = b;
    myglobal.f = f; 

    return Py_BuildValue("i", ok);
}

static PyObject *
justtest_multiply(PyObject *self, PyObject *args) {
    int ok; double coeff;
    ok = PyArg_ParseTuple(args, "d", &coeff);
    myglobal.n = myglobal.n*(int) coeff; 
    myglobal.b = myglobal.b*(float) coeff;
    myglobal.f = myglobal.f*(double) coeff; 

    return Py_BuildValue("i", ok);
}

static PyObject *
justtest_print(PyObject *self, PyObject *args) {
    printf("%i, %f, %e\n", myglobal.n, myglobal.b, myglobal.f);
    return Py_BuildValue("i", 1);
}

static PyMethodDef JusttestMethods[] = {
    {"init",  justtest_init, METH_VARARGS,
     "Initialize global truct"},
    {"multiply",  justtest_multiply, METH_VARARGS,
     "Multiply global truct"},
    {"print",  justtest_print, METH_VARARGS,
     "print global struct"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef justtestmodule = {
    PyModuleDef_HEAD_INIT,
    "justtest",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    JusttestMethods
};


PyMODINIT_FUNC
PyInit_justtest(void)
{
    return PyModule_Create(&justtestmodule);
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
    if (PyImport_AppendInittab("justtest", PyInit_justtest) == -1) {
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
    PyObject* pmodule = PyImport_ImportModule("justtest");
    if (!pmodule) {
        PyErr_Print();
        fprintf(stderr, "Error: could not import module 'justtest'\n");
    }

    PyMem_RawFree(program);
    return 0;
}
