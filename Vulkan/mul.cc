#include <iostream>
using namespace std;

static void mul(int *A, int *B, int *C, int M, int K, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int c = 0;
      for (int k = 0; k < K; ++k) {
        c += A[i * K + k] * B[k * N + j];
      }
      C[i * M + j] = c;
    }
  }
}

static void init(int **ptr, int M, int N) {
  int *A = *ptr;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      A[i * N + j] = i;
    }
  }
}

void Print (int *A, int M, int N) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      cout << A[i * N + j] << " ";
    }
    cout << '\n';
  }
}

int main(int argc, char **argv) {
  const int N = 256;
  const int K = 256;
  const int M = 256;
  int *A = new int[M * K];
  int *B = new int[K * N];
  int *C = new int[N * M];
  init(&A, M, K);
  init(&B, K, N);
  init(&C, M, N);
  mul(A, B, C, M, K, N);
#ifdef DEBUG
  cout << "A : " << endl;
  Print (A, M, K);
  cout << "B : " << endl;
  Print (B, K, N);
  cout << "C : " << endl;
  Print(C, M, N);
#endif
  return C[10 + argc];
}
