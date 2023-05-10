#include "fMatrix.h"

#include <math.h>
#include <stdlib.h>

#include <cmath>
#include <iostream>
#include <limits>

#include "fVector.h"

using namespace std;

// Initialize constructor.
fMatrix::fMatrix(int n_rows, int n_cols) {
    rows = n_rows;
    cols = n_cols;
    elem = new Float[rows * cols];
    // ++nMatCount;
}
// Assign constructor.
fMatrix::fMatrix(Float *Array, int n_rows, int n_cols) {
    rows = n_rows;
    cols = n_cols;
    elem = new Float[rows * cols];
    for (int i = 0; i < rows * cols; ++i) {
        elem[i] = Array[i];
    }
    // ++nMatCount;
}
fMatrix::fMatrix(int n_rows, int n_cols, Float *Array) {
    // 分配存儲空間
    elem = new Float[n_rows * n_cols];
    rows = n_rows;
    cols = n_cols;

    // 將 Array 中的值複製到 elem 中
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            elem[i * cols + j] = Array[i * cols + j];
        }
    }
}
// Copy constructor.
fMatrix::fMatrix(const fMatrix &copy) {
    rows = copy.rows;
    cols = copy.cols;
    elem = new Float[rows * cols];
    for (int i = 0; i < rows * cols; ++i) {
        elem[i] = copy.elem[i];
    }
    // ++nMatCount;
}
// Destructor
fMatrix::~fMatrix() {
    delete[] elem;
    // --nMatCount;
}

// 6. A=B
fMatrix &fMatrix::operator=(const fMatrix &M) {
    if (this != &M) {
        delete[] elem;
        rows = M.rows;
        cols = M.cols;
        elem = new Float[rows * cols];
        for (int i = 0; i < rows * cols; ++i) {
            elem[i] = M.elem[i];
        }
    }
    return *this;
}

// 7. Swap
fMatrix &fMatrix::SwapRows(int i1, int i2) {
    if (i1 >= 0 && i1 < rows && i2 >= 0 && i2 < rows && i1 != i2) {
        for (int j = 0; j < cols; ++j) {
            Float temp = elem[i1 * cols + j];
            elem[i1 * cols + j] = elem[i2 * cols + j];
            elem[i2 * cols + j] = temp;
        }
    }
    return *this;
}
fMatrix &fMatrix::SwapCols(int j1, int j2) {
    if (j1 >= 0 && j1 < cols && j2 >= 0 && j2 < cols && j1 != j2) {
        for (int i = 0; i < rows; ++i) {
            Float temp = elem[i * cols + j1];
            elem[i * cols + j1] = elem[i * cols + j2];
            elem[i * cols + j2] = temp;
        }
    }
    return *this;
}

// Create an nSize x nSize identity matrix.
fMatrix Identity(int nSize) {
    fMatrix I(nSize, nSize);
    for (int i = 0; i < nSize; ++i) {
        I.elem[i * nSize + i] = 1.0;
    }
    return I;
}

// 8. Inverse
fMatrix &fMatrix::Inv(void) {
    if (rows != cols) {
        std::cerr << "Error: cannot invert a non-square matrix." << std::endl;
        return *this;
    }

    fMatrix A(*this);
    fMatrix B = Identity(rows);

    int n = rows;
    for (int i = 0; i < n; ++i) {
        int pivot = i;
        Float pivot_val = A.elem[pivot * cols + i];
        for (int j = i + 1; j < n; ++j) {
            Float temp = fabs(A.elem[j * cols + i]);
            if (temp > pivot_val) {
                pivot_val = temp;
                pivot = j;
            }
        }
        if (pivot_val == 0.0) {
            std::cerr << "Error: singular matrix." << std::endl;
            return *this;
        }
        if (pivot != i) {
            A.SwapRows(i, pivot);
            B.SwapRows(i, pivot);
        }
        Float pivot_inv = 1.0 / A.elem[i * cols + i];
        for (int j = 0; j < n; ++j) {
            A.elem[i * cols + j] *= pivot_inv;
            B.elem[i * cols + j] *= pivot_inv;
        }
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                Float factor = A.elem[j * cols + i];
                for (int k = 0; k < n; ++k) {
                    A.elem[j * cols + k] -= factor * A.elem[i * cols + k];
                    B.elem[j * cols + k] -= factor * B.elem[i * cols + k];
                }
            }
        }
    }

    *this = B;
    return *this;
}
// Get col vector
fVector fMatrix::GetCol(int col) const {
    if (col >= 0 && col < cols) {
        fVector colVec(rows);
        for (int i = 0; i < rows; ++i) {
            colVec(i) = elem[i * cols + col];
        }
        return colVec;
    } else {
        // Return an empty matrix if the column index is out of range
        return fVector();
    }
}
// Get row vector
fVector fMatrix::GetRow(int row) const {
    if (row >= 0 && row < rows) {
        fVector rowVec(cols);
        for (int j = 0; j < cols; ++j) {
            rowVec(j) = elem[row * cols + j];
        }
        return rowVec;
    } else {
        // Return an empty matrix if the row index is out of range
        return fVector();
    }
}
// Get elem
Float fMatrix::Getelem(int col, int row) const {
    if (col < 0 || col >= cols || row < 0 || row >= rows) {
        // Print a error message and show the expected col, row and the actual
        // col, row
        std::cerr << "Index out of range: (" << col << ", " << row << ") in a "
                  << cols << "x" << rows << " matrix." << std::endl;

        // std::cerr << "Index out of range" << std::endl;
        exit(1);
    }
    return elem[row * cols + col];
}

void fMatrix::Show() const {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << elem[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

// 1.A+B
fMatrix operator+(const fMatrix &A, const fMatrix &B) {
    // 先檢查 A 和 B 是否大小一致
    if (A.rows != B.rows || A.cols != B.cols) {
        throw std::invalid_argument("Matrix dimensions must agree.");
    }

    // 創建一個和 A、B 大小相同的新矩陣 C
    fMatrix C(A.rows, A.cols);

    // 將 A、B 的對應元素相加，存儲到 C 中
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            C.elem[i * A.cols + j] =
                A.elem[i * A.cols + j] + B.elem[i * A.cols + j];
        }
    }

    return C;
}

// A-B
fMatrix operator-(const fMatrix &A) {
    // 創建一個和 A 大小相同的新矩陣 B
    fMatrix B(A.rows, A.cols);

    // 將 A 的每個元素取負，存儲到 B 中
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            B.elem[i * A.cols + j] = -A.elem[i * A.cols + j];
        }
    }

    return B;
}
fMatrix operator-(const fMatrix &A, const fMatrix &B) {
    // 先檢查 A 和 B 是否大小一致
    if (A.rows != B.rows || A.cols != B.cols) {
        throw std::invalid_argument("Matrix dimensions must agree.");
    }

    // 創建一個和 A、B 大小相同的新矩陣 C
    fMatrix C(A.rows, A.cols);

    // 將 A、B 的對應元素相減，存儲到 C 中
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            C.elem[i * A.cols + j] =
                A.elem[i * A.cols + j] - B.elem[i * A.cols + j];
        }
    }

    return C;
}

// A*c or c*A
fMatrix operator*(const fMatrix &A, Float s) {
    // 創建一個和 A 大小相同的新矩陣 B
    fMatrix B(A.rows, A.cols);

    // 將 A 的每個元素乘以 s，存儲到 B 中
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            B.elem[i * A.cols + j] = A.elem[i * A.cols + j] * s;
        }
    }

    return B;
}
fMatrix operator*(Float s, const fMatrix &A) {
    // 創建一個和 A 大小相同的新矩陣 B
    fMatrix B(A.rows, A.cols);

    // 將 A 的每個元素乘以 s，存儲到 B 中
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            B.elem[i * A.cols + j] = A.elem[i * A.cols + j] * s;
        }
    }

    return B;
}

// A/c
fMatrix operator/(const fMatrix &A, Float s) {
    // 創建一個和 A 大小相同的新矩陣 B
    fMatrix B(A.rows, A.cols);

    // 將 A 的每個元素除以 s，存儲到 B 中
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            B.elem[i * A.cols + j] = A.elem[i * A.cols + j] / s;
        }
    }

    return B;
}

// A*B
fMatrix operator*(const fMatrix &A, const fMatrix &B) {
    // 先檢查 A 和 B 是否可以相乘
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimensions must agree.");
    }

    // 創建一個和 A 的行數、B 的列數相同的新矩陣 C
    fMatrix C(A.rows, B.cols);

    // 對 C 的每個元素進行計算
    for (int i = 0; i < C.rows; i++) {
        for (int j = 0; j < C.cols; j++) {
            Float sum = 0;
            for (int k = 0; k < A.cols; k++) {
                sum += A.elem[i * A.cols + k] * B.elem[k * B.cols + j];
            }
            C.elem[i * C.cols + j] = sum;
        }
    }

    return C;
}

// operation
fMatrix &operator+=(fMatrix &A, const fMatrix &B) {
    // 先檢查 A 和 B 是否大小相同
    if (A.rows != B.rows || A.cols != B.cols) {
        throw std::invalid_argument("Matrix dimensions must agree.");
    }

    // 對 A 的每個元素進行計算
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            A.elem[i * A.cols + j] += B.elem[i * B.cols + j];
        }
    }

    return A;
}
fMatrix &operator-=(fMatrix &A, const fMatrix &B) {
    // 先檢查 A 和 B 是否大小相同
    if (A.rows != B.rows || A.cols != B.cols) {
        throw std::invalid_argument("Matrix dimensions must agree.");
    }

    // 對 A 的每個元素進行計算
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            A.elem[i * A.cols + j] -= B.elem[i * B.cols + j];
        }
    }

    return A;
}
fMatrix &operator*=(fMatrix &A, Float s) {
    // 對 A 的每個元素進行計算
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            A.elem[i * A.cols + j] *= s;
        }
    }

    return A;
}
fMatrix &operator*=(fMatrix &A, const fMatrix &B) {
    // 先檢查 A 和 B 是否可以相乘
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimensions must agree.");
    }

    // 創建一個和 A 的行數、B 的列數相同的新矩陣 C
    fMatrix C(A.rows, B.cols);

    // 對 C 的每個元素進行計算
    for (int i = 0; i < C.rows; i++) {
        for (int j = 0; j < C.cols; j++) {
            Float sum = 0;
            for (int k = 0; k < A.cols; k++) {
                sum += A.elem[i * A.cols + k] * B.elem[k * B.cols + j];
            }
            C.elem[i * C.cols + j] = sum;
        }
    }

    // 將 C 的值賦給 A
    A = C;

    return A;
}
fMatrix &operator/=(fMatrix &A, Float s) {
    // 對 A 的每個元素進行計算
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < A.cols; j++) {
            A.elem[i * A.cols + j] /= s;
        }
    }

    return A;
}

// Transpose of a matrix
fMatrix Transp(const fMatrix &A) {
    // 創建一個新的 fMatrix 對象 B，其行數和列數與 A 相反
    fMatrix B(A.cols, A.rows);

    // 對 B 的每個元素進行計算
    for (int i = 0; i < B.rows; i++) {
        for (int j = 0; j < B.cols; j++) {
            B.elem[i * B.cols + j] = A.elem[j * A.cols + i];
        }
    }

    return B;
}
// Computes A * Transp(A).
fMatrix AATransp(const fMatrix &A) {
    // 創建一個新的 fMatrix 對象 B，其行數和列數均為 A 的行數
    fMatrix B(A.rows, A.rows);

    // 對 B 的每個元素進行計算
    for (int i = 0; i < B.rows; i++) {
        for (int j = 0; j < B.rows; j++) {
            Float sum = 0;
            for (int k = 0; k < A.cols; k++) {
                sum += A.elem[i * A.cols + k] * A.elem[j * A.cols + k];
            }
            B.elem[i * B.cols + j] = sum;
        }
    }

    return B;
}
// Computes Transp(A) * A.
fMatrix ATranspA(const fMatrix &A) {
    // 創建一個新的 fMatrix 對象 B，其行數和列數均為 A 的列數
    fMatrix B(A.cols, A.cols);

    // 對 B 的每個元素進行計算
    for (int i = 0; i < B.rows; i++) {
        for (int j = 0; j < B.cols; j++) {
            Float sum = 0;
            for (int k = 0; k < A.rows; k++) {
                sum += A.elem[k * A.cols + i] * A.elem[k * A.cols + j];
            }
            B.elem[i * B.cols + j] = sum;
        }
    }

    return B;
}

// Computes the determinant of a square matrix
double Determinant(const fMatrix &A) {
    // 如果 A 是 1 x 1 的矩陣，則行列式等於它的唯一元素
    if (A.rows == 1) {
        return A.elem[0];
    }

    // 如果 A 是 2 x 2 的矩陣，則行列式等於 a11*a22 - a12*a21
    if (A.rows == 2) {
        return A.elem[0] * A.elem[3] - A.elem[1] * A.elem[2];
    }

    // 創建一個新的 fMatrix 對象 B，為 A 的一個子矩陣
    fMatrix B(A.rows - 1, A.cols - 1);

    // 定義一個變量 sign，用於表示每個子矩陣的符號
    int sign = 1;

    // 定義一個變量 det，用於存儲行列式的值
    double det = 0;

    // 對 A 的第一行進行循環，計算每個子矩陣的行列式並加總
    for (int j = 0; j < A.cols; j++) {
        // 構造子矩陣 B
        for (int i = 1; i < A.rows; i++) {
            for (int k = 0; k < A.cols; k++) {
                if (k < j) {
                    B.elem[(i - 1) * B.cols + k] = A.elem[i * A.cols + k];
                } else if (k > j) {
                    B.elem[(i - 1) * B.cols + (k - 1)] = A.elem[i * A.cols + k];
                }
            }
        }

        // 遞歸計算子矩陣 B 的行列式
        det += sign * A.elem[j] * Determinant(B);

        // 改變符號
        sign = -sign;
    }

    return det;
}
// Computes the trace of a square matrix
double Trace(const fMatrix &A) {
    // 計算 A 的迹
    double tr = 0;
    for (int i = 0; i < A.rows; i++) {
        tr += A.elem[i * A.cols + i];
    }

    return tr;
}
// Computes the L1-norm of the matrix A, which is the maximum absolute column
// sum.
double OneNorm(const fMatrix &A) {
    // 計算 A 的 L1-范數
    double norm = 0;
    for (int j = 0; j < A.cols; j++) {
        double col_sum = 0;
        for (int i = 0; i < A.rows; i++) {
            col_sum += std::abs(A.elem[i * A.cols + j]);
        }
        norm = std::max(norm, col_sum);
    }

    return norm;
}
// Computes the Inf-norm of the matrix A, which is the maximum absolute row sum.
double InfNorm(const fMatrix &A) {
    // 計算 A 的無窮范數
    double norm = 0;
    for (int i = 0; i < A.rows; i++) {
        double row_sum = 0;
        for (int j = 0; j < A.cols; j++) {
            row_sum += std::abs(A.elem[i * A.cols + j]);
        }
        norm = std::max(norm, row_sum);
    }

    return norm;
}
// Computes the inverse of a square matrix.
fMatrix Inverse(const fMatrix &A) {
    // 創建一個新的 fMatrix 對象 B，初始化為 A
    fMatrix B(A);

    // 創建一個新的 fMatrix 對象 I，作為恆等矩陣
    fMatrix I(A.rows, A.cols);
    for (int i = 0; i < A.rows; i++) {
        I.elem[i * A.cols + i] = 1;
    }

    // 利用高斯-喬丹消元法計算 A 的逆矩陣
    for (int j = 0; j < A.cols; j++) {
        // 將 B 的第 j 列與 I 的第 j 列交換，以確保對角線元素不為零
        int max_idx = j;
        double max_val = std::abs(B.elem[j * A.cols + j]);
        for (int i = j + 1; i < A.rows; i++) {
            double val = std::abs(B.elem[i * A.cols + j]);
            if (val > max_val) {
                max_idx = i;
                max_val = val;
            }
        }
        if (max_idx != j) {
            B.SwapRows(j, max_idx);
            I.SwapRows(j, max_idx);
        }

        // 將 B 的第 j 列除以 B(j,j)，以使 B(j,j) 等於 1
        double pivot = B.elem[j * A.cols + j];
        for (int k = 0; k < A.cols; k++) {
            B.elem[j * A.cols + k] /= pivot;
            I.elem[j * A.cols + k] /= pivot;
        }

        // 將 B 的其它行減去 B 的第 j 行的適當倍數，以使 B(i,j) 等於 0（i≠j）
        for (int i = 0; i < A.rows; i++) {
            if (i == j) {
                continue;
            }
            double factor = B.elem[i * A.cols + j];
            for (int k = 0; k < A.cols; k++) {
                B.elem[i * A.cols + k] -= factor * B.elem[j * A.cols + k];
                I.elem[i * A.cols + k] -= factor * I.elem[j * A.cols + k];
            }
        }
    }

    return I;
}

fMatrix Diag(fVector &v) {
    int n = v.Size();
    fMatrix mat(n, n);
    for (int i = 0; i < n; i++) {
        mat.Setelem(i, i, v(i));
    }
    return mat;
}

void fMatrix::SetSize(int rows, int cols) {
    if (rows < 0) rows = 0;
    if (cols < 0) cols = 0;

    if (rows != this->rows || cols != this->cols) {
        if (elem) {
            delete[] elem;
            elem = NULL;
        }
        if (rows * cols > 0) {
            elem = new Float[rows * cols];
            for (int i = 0; i < rows * cols; i++) {
                elem[i] = 0.0;
            }
        }
    }

    this->rows = rows;
    this->cols = cols;
}

void fMatrix::SetIdentity() {
    for (int i = 0; i < Rows(); ++i) {
        for (int j = 0; j < Cols(); ++j) {
            if (i == j) {
                Setelem(i, j, 1.0);
            } else {
                Setelem(i, j, 0.0);
            }
        }
    }
}

fMatrix GivensRotation(int i, int j, Float theta, int n) {
    fMatrix G(n, n);
    G.SetIdentity();
    G.Setelem(i, i, cos(theta));
    G.Setelem(j, j, cos(theta));
    G.Setelem(i, j, -sin(theta));
    G.Setelem(j, i, sin(theta));
    return G;
}

Float TwoNorm(const fVector &v) {
    Float norm = 0.0;
    for (int i = 0; i < v.Size(); ++i) {
        norm += v(i) * v(i);
    }
    return sqrt(norm);
}

void SVDcmp(fMatrix &AU, fVector &W, fMatrix &V) {
    int m = AU.Rows();
    int n = AU.Cols();
    fMatrix U(m, m), V_temp(n, n);
    U.SetIdentity();
    V_temp.SetIdentity();

    Float tol = 1e-10;
    bool converged = false;

    int max_iterations = 100;  // Set a maximum number of iterations
    int iteration = 0;         // Initialize the iteration counter

    while (!converged && iteration < max_iterations) {
        converged = true;
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                Float a = 0, b = 0, c = 0;
                for (int k = 0; k < m; k++) {
                    a += AU.Getelem(k, i) * AU.Getelem(k, i);
                    b += AU.Getelem(k, j) * AU.Getelem(k, j);
                    c += AU.Getelem(k, i) * AU.Getelem(k, j);
                }
                if (fabs(c) > tol * sqrt(a * b)) {
                    converged = false;
                    Float theta = 0.5 * atan2(2 * c, a - b);
                    fMatrix G = GivensRotation(
                        i, j, theta, m);  // Change the size of the Givens
                                          // rotation matrix to 'm'
                    AU = AU * G;
                    U = U * G;
                    V_temp = V_temp * Transp(G);
                }
            }
        }
        iteration++;
    }

    for (int i = 0; i < n; i++) {
        W(i) = TwoNorm(AU.GetCol(i));
    }

    // Copy the results to the input matrices
    AU = U;
    V = V_temp;

    // Print the resulting matrix
    // AU.Show();
}

fMatrix Cholesky(const fMatrix &A) {
    int n = A.Rows();
    if (n != A.Cols()) {
        throw std::invalid_argument(
            "Input matrix must be square for Cholesky decomposition.");
    }

    fMatrix L(n, n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            Float sum = 0;
            if (j == i) {
                for (int k = 0; k < j; k++) {
                    sum += L.Getelem(j, k) * L.Getelem(j, k);
                }
                L.Setelem(j, j, sqrt(A.Getelem(j, j) - sum));
            } else {
                for (int k = 0; k < j; k++) {
                    sum += L.Getelem(i, k) * L.Getelem(j, k);
                }
                L.Setelem(i, j, (A.Getelem(i, j) - sum) / L.Getelem(j, j));
            }
        }
    }

    return L;
}
