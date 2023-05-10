#include <iostream>

#include "fMatrix.h"
#include "fVector.h"

using namespace std;

void testMatrixFuns(void) {
    Float A[3] = {1.1, 2.2, 3.3};
    Float B[3] = {4.4, 5.5, 6.6};
    Float C[9] = {0.9649, 0.9572, 0.1419, 0.1576, 0.4854,
                  0.4218, 0.9706, 0.8003, 0.9157};
    Float D[9] = {0.8147, 0.9134, 0.2785, 0.9058, 0.6324,
                  0.5469, 0.1270, 0.0975, 0.9575};
    Float X[15] = {0.7922, 0.9340, 0.6555, 0.9595, 0.6787,
                   0.1712, 0.6557, 0.7577, 0.7060, 0.0357,
                   0.7431, 0.0318, 0.8491, 0.3922, 0.2769};
    fVector VecA(A, 3);
    fVector VecB(B, 3);
    fVector VecC(VecA);
    fMatrix MatA(C, 3, 3);
    fMatrix MatB(D, 3, 3);
    fMatrix MatC(MatA);
    fMatrix MatX(X, 3, 3);
    fMatrix MatXt = Transp(MatX);

    // Declare and initialize the necessary variables and objects
    cout << "Basic functionality test." << endl;
    // fVector VecA(A, 3);            // Declare a vector VecA.
    // fMatrix MatA(C, 3, 3);         // Declare a matrix MatA.
    cout << VecA(1) << endl;      // Get the first element of VecA.
    cout << VecA.Size() << endl;  // Get the size of VecA.
    cout << MatA.Cols() << endl;  // Get the number of columns of MatA.
    cout << MatA.Rows() << endl;  // Get the number of rows of MatA.

    // Print out the matrices for visual inspection
    cout << "\nMatA = " << endl;
    MatA.Show();

    cout << "\nMatB = " << endl;
    MatB.Show();

    cout << "\nMatX = " << endl;
    MatX.Show();

    // Begin testing matrix operators
    cout << "\nStarts to test matrix operators..." << endl;

    // Test operator A+B
    cout << "\n1. A+B" << endl;
    (MatA + MatB).Show();

    // Test operator A-B
    cout << "\n2. A-B" << endl;
    (MatA - MatB).Show();

    // Test operator A*B
    cout << "\n3. A*B" << endl;
    (MatA * MatB).Show();

    // Test operator 2*A
    cout << "\n4. 2*A" << endl;
    (2 * MatA).Show();

    // Test operator A/2
    cout << "\n5. A/2" << endl;
    (MatA / 2).Show();

    // Test swapping rows in matrix A
    cout << "row 1 to row 3" << endl;
    (MatA).Show();
    cout << endl;
    (MatA.SwapRows(0, 2)).Show();

    // Test getting row 1 of matrix A
    cout << "get row 1" << endl;
    (MatA).Show();
    cout << endl;
    (MatA.GetRow(0)).Show();
    cout << endl;

    // Test getting column 2 of matrix A
    cout << "get cols 2" << endl;
    (MatA).Show();
    cout << endl;
    (MatA.GetCol(1)).Show();
    cout << endl;

    // Test getting an element in matrix A
    cout << "get (0,1))" << endl;
    (MatA).Show();
    cout << MatA.Getelem(0, 1) << endl;

    // Test transposing matrix A
    cout << "Transpose" << endl;
    (MatA).Show();
    cout << endl;
    (Transp(MatA)).Show();
    cout << endl;

    // Test calculating the determinant of matrix A
    cout << "Determinent" << endl;
    (MatA).Show();
    cout << endl;
    cout << Determinant(MatA) << endl;

    // Test calculating the inverse of matrix A
    cout << "Inverse" << endl;
    (MatA).Show();
    cout << endl;
    (Inverse(MatA)).Show();
    cout << endl;

    // Test multiplying matrix A by its inverse
    (MatA * Inverse(MatA)).Show();

    cout << "SVD decomposition functionality test." << endl;
    fMatrix MatU(MatX);
    fVector VecD(MatX.Cols());
    fMatrix MatV(MatX.Cols(), MatX.Cols());
    SVDcmp(MatU, VecD, MatV);

    cout << "\nMatU" << endl;
    MatU.Show();
    cout << "\nMatD" << endl;
    // Diag(VecD).Show();
    VecD.Show();
    cout << "\nMatV" << endl;
    MatV.Show();
    cout << endl << endl;

    cout << "Cholesky decomposition functionality test." << endl;
    fMatrix MatL = Cholesky(MatX);
    cout << "\nMatL" << endl;
    MatL.Show();

    return;
}

int main(void) {
    testMatrixFuns();
    return 0;
}