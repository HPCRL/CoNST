#include "taco.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sys/stat.h>
#include <taco/storage/file_io_tns.h>
#include <vector>

using namespace taco;

const int II = 183;
const int J = 24;
const int K = 1140;
const int L = 1717;
const int A = 50;
const int B = 50;
const int C = 50;

void ttmc_unfused(Tensor<double> I, Tensor<double> M1, Tensor<double> M2,
                  Tensor<double> R, Tensor<double> M3) {
  IndexVar l("l");
  IndexVar k("k");
  IndexVar a("a");
  IndexVar b("b");
  IndexVar j("j");
  IndexVar c("c");
  IndexVar i("i");
  // Tensor<double> Im1("IM1", {I, A, L, K}, {Dense, Sparse, Sparse, Sparse});
  Tensor<double> IM1 =
      Tensor<double>("IM1", {II, A, L, K}, {Dense, Sparse, Sparse, Sparse});

  Tensor<double> IM1M2 =
      Tensor<double>("IM1M2", {II, A, B, L}, {Dense, Sparse, Sparse, Sparse});
  IM1(i, a, l, k) = I(i, l, k, j) * M1(a, j);
  IM1M2(i, a, b, l) = IM1(i, a, l, k) * M2(b, k);
  R(i, a, b, c) += IM1M2(i, a, b, l) * M3(c, l);

  IM1.compile();
  IM1M2.compile();
  R.compile();
  auto start = std::chrono::high_resolution_clock::now();
  IM1.assemble();
  IM1M2.assemble();
  IM1.compute();
  IM1M2.compute();
  R.assemble();
  R.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time ttmc_unfused:  " << elapsed.count() << " ms " << std::endl;
  write("ttmc_unfused.tns", R);
}


int main() {
  Tensor<double> I1("input", {II, J, K, L}, {Dense, Sparse, Sparse, Sparse});
  I1 = read("downloads/uber.tns", {Dense, Sparse, Sparse, Sparse});
  I1.pack();
  I1 = I1.transpose({0, 3, 2, 1});
  I1.setName("Input");
  std::cout << "Read I" << std::endl;

  Tensor<double> M1("M1", {J, A}, {Dense, Dense});
  M1 = read("m1.tns", {Dense, Dense});
  M1 = M1.transpose({1, 0});
  M1.pack();
  std::cout << "Read M1" << std::endl;

  Tensor<double> M2("M2", {K, B}, {Dense, Dense});
  M2 = read("m2.tns", {Dense, Dense});
  M2 = M2.transpose({1, 0});
  M2.pack();
  std::cout << "Read M2" << std::endl;

  Tensor<double> M3("M3", {L, C}, {Dense, Dense});
  M3 = read("m3.tns", {Dense, Dense});
  M3 = M3.transpose({1, 0});
  M3.pack();
  std::cout << "Read all data" << std::endl;

  Tensor<double> R("result", {II, A, B, C}, {Dense, Dense, Dense, Dense});
  ttmc_unfused(I1, M1, M2, R, M3);

  return 0;
}
