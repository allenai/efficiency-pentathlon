#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>

int rprof(int argc, char ** argv);

static PyMethodDef Methods[] = {
    {"rprof", rprof, METH_VARARGS, "efficiency"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef Module = {
    PyModuleDef_HEAD_INIT, "Module",
    "Module", -1, Methods
};

PyMODINIT_FUNC PyInit_Module(void) {
    return PyModule_Create(&Module);
}