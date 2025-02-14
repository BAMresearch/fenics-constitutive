#pragma once
#include <dlfcn.h>
#include <string>
#include <iostream>

namespace constants
{
// constants have internal linkage by default
const int ntens = 6;
const int dim = 3;
} // namespace constants

typedef void (*t_Param)(char[], int);
typedef void (*t_Eval)(double[], double[], double (*ddsdde)[constants::ntens], double*, double*, double*, double*,
                       double[], double[], double*, double[], double[], double[], double*, double*, double*, double*,
                       double*, char[], int*, int*, int*, int*, double[], int*, double[constants::dim],
                       double (*drot)[constants::dim], double*, double*, double (*dfgrd0)[constants::dim],
                       double (*dfgrd1)[constants::dim], int*, int*, int*, int*, int*, int*, int);

class LibHandle
{
public:
    LibHandle(std::string libName)
        : _libName(libName)
    {
        if (access(libName.c_str(), F_OK) == -1)
        {
            throw std::runtime_error("Library at " + libName + " does not exist!");
        }
        // load a shared fortran library
        _libHandle = dlopen(libName.c_str(), RTLD_LAZY);
        if (!_libHandle)
        {
            std::cerr << dlerror() << std::endl;
            dlclose(_libHandle);
            throw std::runtime_error("Cannot load library " + libName + "!");
        }
    }

    template <typename T>
    T Load(std::string name)
    {
        T f = (T)dlsym(_libHandle, name.c_str());
        if (!f)
        {
            std::cerr << dlerror() << std::endl;
            throw std::runtime_error("Cannot load function " + name + " from " + _libName + "!");
        }
        return f;
    }

    ~LibHandle()
    {
        dlclose(_libHandle);
    }

private:
    void* _libHandle;
    std::string _libName;
};

