/*
    Copyright (C) 2014-2020, Johannes Pekkila, Miikka Vaisala.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * @file
 * \brief Brief info.
 *
 * Detailed info.
 *
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <astaroth.h>

static PyObject *
astaroth_acDeviceCreate(PyObject *self, PyObject *args) {


    return Py_BuildValue("i", 1);
}

static PyObject *
astaroth_acDevicePrintInfo(PyObject *self, PyObject *args) {


    return Py_BuildValue("i", 1);
}
static PyObject *
astaroth_acDeviceDestroy(PyObject *self, PyObject *args) {


    return Py_BuildValue("i", 1);
}

static PyMethodDef AstarothMethods[] = {
    {"acDeviceCreate",    astaroth_acDeviceCreate, METH_VARARGS,
     "acDeviceCreate"},
    {"acDevicePrintInfo", astaroth_acDevicePrintInfo, METH_VARARGS,
     "acDevicePrintInfo"},                 
    {"acDeviceDestroy",   astaroth_acDeviceDestroy, METH_VARARGS,
     "acDeviceDestroy"},
    {NULL, NULL, 0, NULL}       
};

static struct PyModuleDef astarothmodule = {
    PyModuleDef_HEAD_INIT,
    "astaroth", 
    "Add Astaroth documentation line here.",      
    -1,        
    AstarothMethods
};


PyMODINIT_FUNC
PyInit_mpitest(void)
{
    return PyModule_Create(&astarothmodule);
}

int
main(int argc, char *argv[])
{
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    if (PyImport_AppendInittab("astaroth", PyInit_mpitest) == -1) {
        fprintf(stderr, "Error: could not extend in-built modules table\n");
        exit(1);
    }

    Py_SetProgramName(program);

    Py_Initialize();

    PyObject* pmodule = PyImport_ImportModule("astaroth");
    if (!pmodule) {
        PyErr_Print();
        fprintf(stderr, "Error: could not import module 'mpitest'\n");
    }

    PyMem_RawFree(program);
    return 0;
}
