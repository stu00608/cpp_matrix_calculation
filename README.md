# C++ Matrix and Vector Manipulation Project

## Introduction

This C++ project aims to provide a comprehensive library for matrix and vector manipulation, as well as implementing advanced mathematical algorithms such as Singular Value Decomposition (SVD), Cholesky decomposition, and finding homographies. The project is divided into several files, each containing the implementation of specific classes and functions.

## Files and Classes

1. `fVector.cpp`: This file contains the implementation of the `fVector` class, which represents a mathematical vector and provides basic vector operations.

2. `fMatrix.cpp`: This file contains the implementation of the `fMatrix` class, which represents a mathematical matrix and provides basic matrix operations, including addition, subtraction, multiplication, and division. It also includes advanced matrix operations, such as matrix inversion, transpose, and determinant calculation.

3. `MatchingPoints.cpp`: This file contains the implementation of the `findHomography` function, which calculates the homography matrix between two sets of points.

4. `main.cpp`: This file contains the main function, which demonstrates the usage of the library and tests various matrix and vector operations.

## How to Run

To run the project, compile all the source files using a C++ compiler, such as `g++`, and then execute the generated binary file. For example:

```bash
g++ fVector.cpp fMatrix.cpp MatchingPoints.cpp main.cpp -o matrix_vector
./matrix_vector
```

## Usage

To use the library in your own project, include the header files `fVector.h` and `fMatrix.h` in your source code. Then, create instances of the `fVector` and `fMatrix` classes and use their member functions to perform various matrix and vector operations.

### Example

Here's an example of how to use the library to perform matrix and vector operations:

```cpp
#include <iostream>
#include "fVector.h"
#include "fMatrix.h"

int main() {
    // Create a vector with 3 elements
    fVector vecA(3);
    vecA(0) = 1.0;
    vecA(1) = 2.0;
    vecA(2) = 3.0;

    // Create a 3x3 matrix
    fMatrix matA(3, 3);
    matA.Setelem(0, 0, 1.0);
    matA.Setelem(0, 1, 2.0);
    matA.Setelem(0, 2, 3.0);
    matA.Setelem(1, 0, 4.0);
    matA.Setelem(1, 1, 5.0);
    matA.Setelem(1, 2, 6.0);
    matA.Setelem(2, 0, 7.0);
    matA.Setelem(2, 1, 8.0);
    matA.Setelem(2, 2, 9.0);

    // Multiply the matrix by a scalar
    fMatrix matB = matA * 2;

    // Display the result
    std::cout << "Matrix B:" << std::endl;
    matB.Show();

    return 0;
}
```

## Conclusion

This C++ project provides a comprehensive library for matrix and vector manipulation, as well as advanced mathematical algorithms such as SVD, Cholesky decomposition, and finding homographies. By using this library, developers can easily perform various matrix and vector operations in their projects. The provided example demonstrates the basic usage of the library, and the main function in `main.cpp` serves as a test suite for the implemented functionality.
