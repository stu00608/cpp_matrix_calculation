#include "fVector.h"

#include <math.h>
#include <stdlib.h>

#include <iostream>

#include "fMatrix.h"

using namespace std;

///**********************Vector**************////////////
// Initinalize constructor.
fVector::fVector(int size) {
    this->size = size;
    if (size > 0) {
        elem = new Float[size];
        for (int i = 0; i < size; i++) {
            elem[i] = 0;
        }
    } else {
        elem = nullptr;
    }
}
// Copy constructor.
fVector::fVector(const fVector &other) {
    size = other.size;
    if (size > 0) {
        elem = new Float[size];
        for (int i = 0; i < size; i++) {
            elem[i] = other.elem[i];
        }
    } else {
        elem = nullptr;
    }
}
// Assign constructor.
fVector::fVector(const Float *x, int n) {
    size = n;
    if (size > 0) {
        elem = new Float[size];
        for (int i = 0; i < size; i++) {
            elem[i] = x[i];
        }
    } else {
        elem = nullptr;
    }
}
fVector::fVector(int n, const Float *x) {
    size = n;
    if (size > 0) {
        elem = new Float[size];
        for (int i = 0; i < size; i++) {
            elem[i] = x[i];
        }
    } else {
        elem = nullptr;
    }
}
fVector::fVector(Float x, Float y) {
    size = 2;
    elem = new Float[2];
    elem[0] = x;
    elem[1] = y;
}
fVector::fVector(Float x, Float y, Float z) {
    size = 3;
    elem = new Float[3];
    elem[0] = x;
    elem[1] = y;
    elem[2] = z;
}
// Destructor.
fVector::~fVector() {
    if (elem != nullptr) {
        delete[] elem;
        elem = nullptr;
    }
}

void fVector::Show(VecType Type) const {
    if (Type == RowVec) {
        for (int i = 0; i < size; i++) {
            std::cout << elem[i] << " ";
        }
        std::cout << std::endl;
    } else  // ColVec
    {
        for (int i = 0; i < size; i++) {
            std::cout << elem[i] << std::endl;
        }
    }
}