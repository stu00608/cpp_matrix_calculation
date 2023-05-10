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
    fMatrix MatX(X, 5, 3);
    fMatrix MatXt = Transp(MatX);

    cout << "\nMatA = " << endl;
    MatA.Show();

    cout << "\nMatB = " << endl;
    MatB.Show();

    cout << "\nMatX = " << endl;
    MatX.Show();

    cout << "\nStarts to test matrix operators..." << endl;
    // 1. A+B
    cout << "\n1. A+B" << endl;
    (MatA + MatB).Show();

    // 2. A-B
    cout << "\n2. A-B" << endl;
    (MatA - MatB).Show();

    // 4. 2*A
    cout << "\n4. 2*A" << endl;
    (2 * MatA).Show();

    // 6. A/2
    cout << "\n6. A/2" << endl;
    (MatA / 2).Show();

    // 7. A*B
    cout << "\n7. A*B" << endl;
    (MatA * MatB).Show();

    // row trans
    cout << "row 1 to row 3" << endl;
    (MatA).Show();
    cout << endl;
    (MatA.SwapRows(0, 2)).Show();

    // Get row
    cout << "get row 1" << endl;
    (MatA).Show();
    cout << endl;
    (MatA.GetRow(0)).Show();
    cout << endl;
    /// Get cols
    cout << "get cols 2" << endl;
    (MatA).Show();
    cout << endl;
    (MatA.GetCol(1)).Show();
    cout << endl;

    // Get a elem
    cout << "get (0,1))" << endl;
    (MatA).Show();
    cout << endl;
    (MatA.Getelem(0, 1)).Show();

    // Transpose Matrix
    cout << "Transpose" << endl;
    (MatA).Show();
    cout << endl;
    (Transp(MatA)).Show();
    cout << endl;

    // Determinent Value
    cout << "Determinent" << endl;
    (MatA).Show();
    cout << endl;
    cout << Determinant(MatA) << endl;

    // Inverse
    cout << "Inverse" << endl;
    (MatA).Show();
    cout << endl;
    (Inverse(MatA)).Show();
    cout << endl;

    (MatA * Inverse(MatA)).Show();
}

int main(void) {
    testMatrixFuns();
    cout << "Hello, world!\n" << endl;
    return 0;
}