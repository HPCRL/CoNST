#include "3c_filter_fused.hpp"
#include "3c_nofilter_fused.hpp"
#include "taco.h"
#include "util.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sys/stat.h>
#include <taco/storage/file_io_tns.h>
#include <vector>

using namespace taco;

void noncov_mp2_unfused(Tensor<double> Int, Tensor<double> C,
                        Tensor<double> Phat, Tensor<double> X,
                        Tensor<double> L) {
  IndexVar nu("nu");
  IndexVar muhat("muhat");
  IndexVar k("k");
  IndexVar mu("mu");
  IndexVar i("i");
  Int = Int.transpose({2, 0, 1});
  C = C.transpose({1, 0});
  Tensor<double> IntC =
      Tensor<double>("IntC", {AUX, MO, PAO}, {Dense, Sparse, Sparse});
  Tensor<double> IntCPhat =
      Tensor<double>("IntCPhat", {PAO_HAT, AUX, MO}, {Dense, Sparse, Sparse});
  IntC(k, i, mu) = Int(k, mu, nu) * C(i, nu);
  IntCPhat(muhat, k, i) = IntC(k, i, mu) * Phat(muhat, mu);
  X(muhat, k, i) = IntCPhat(muhat, k, i) * L(k, i);
  IntC.compile();
  IntCPhat.compile();
  X.compile();
  auto start = std::chrono::high_resolution_clock::now();
  IntC.assemble();
  IntCPhat.assemble();
  X.assemble();
  IntC.compute();
  IntCPhat.compute();
  X.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time filter TACO-unfused:  " << elapsed.count() << " ms "
            << std::endl;
  // write("data_3cent/X_unfused.tns", X);
}

void noncov_mp2_nary(Tensor<double> Int, Tensor<double> C, Tensor<double> Phat,
                     Tensor<double> X, Tensor<double> L) {
  IndexVar nu("nu");
  IndexVar muhat("muhat");
  IndexVar k("k");
  IndexVar mu("mu");
  IndexVar i("i");
  Int = Int.transpose({2, 0, 1});
  C = C.transpose({1, 0});
  Phat = Phat.transpose({1, 0});
  Phat.setName("Phat");

  X(k, i, muhat) = Int(k, nu, mu) * C(i, nu) * Phat(muhat, mu) * L(k, i);
  X.compile();
  auto start = std::chrono::high_resolution_clock::now();
  X.assemble();
  X.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time filter TACO-Nary:  " << elapsed.count() << " ms "
            << std::endl;
  // write("nary_3cint.tns", X);
}

void nofilter_unfused(Tensor<double> Int, Tensor<double> C, Tensor<double> X,
                      Tensor<double> Phat) {
  IndexVar i("i");
  IndexVar mu("mu");
  IndexVar muhat("muhat");
  IndexVar nu("nu");
  IndexVar k("k");
  Int = Int.transpose({2, 0, 1});
  C = C.transpose({1, 0});
  Tensor<double> IntC =
      Tensor<double>("IntC", {AUX, MO, PAO}, {Dense, Sparse, Sparse});
  IntC(k, i, mu) = Int(k, mu, nu) * C(i, nu);
  X(k, i, muhat) = IntC(k, i, mu) * Phat(muhat, mu);
  IntC.compile();
  X.compile();
  auto start = std::chrono::high_resolution_clock::now();
  IntC.assemble();
  IntC.compute();
  X.assemble();
  X.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time nofilter TACO-unfused:  " << elapsed.count() << " ms "
            << std::endl;
  // write("data_3cent/X_nofilter_unf.tns", X);
}

void nofilter_nary(Tensor<double> Int, Tensor<double> C, Tensor<double> X,
                   Tensor<double> Phat) {
  IndexVar i("i");
  IndexVar mu("mu");
  IndexVar muhat("muhat");
  IndexVar nu("nu");
  IndexVar k("k");
  Int = Int.transpose({2, 0, 1});
  C = C.transpose({1, 0});
  Phat = Phat.transpose({1, 0});
  Phat.setName("Phat");
  X(k, i, muhat) = Int(k, nu, mu) * C(i, nu) * Phat(muhat, mu);
  X.compile();

  auto start = std::chrono::high_resolution_clock::now();
  X.assemble();
  X.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time nofilter TACO-Nary:  " << elapsed.count() << " ms "
            << std::endl;
  // write("threec_int_nary.tns", X);
}

int main(int argc, char *argv[]) {
  assert(argc == 2);
  std::string exp_size = argv[1];
  std::string path_prefix = parseExperiment(exp_size);
  // Tensor<double> Int("Int", {PAO, PAO, AUX}, {Dense, Sparse, Sparse});
  Tensor<double> Int = read(path_prefix + "/Int.tns", {Dense, Sparse, Sparse});
  Int.pack();
  // Int = Int.transpose({0, 2, 1});
  Int.setName("Int");

  // Tensor<double> L("L", {AUX, MO}, {Dense, Sparse});
  Tensor<double> L = read(path_prefix + "/L.tns", {Dense, Sparse});
  L.pack();
  L.setName("L");

  Tensor<double> Phat("Phat", {PAO, PAO_HAT}, {Dense, Sparse});
  Phat = read(path_prefix + "/Phat.tns", {Dense, Sparse});
  Phat.pack();
  Phat.setName("Phat");

  Tensor<double> C("C", {PAO, MO}, {Dense, Sparse});
  C = read(path_prefix + "/C.tns", {Dense, Sparse});
  Phat.pack();
  C.setName("C");

  Tensor<double> X("X", {AUX, MO, PAO_HAT}, {Dense, Sparse, Sparse});
  filter_const(Int, C, Phat, X, L);
  //Tensor<double> X_nofilter("X", {MO, AUX, PAO_HAT}, {Dense, Sparse, Sparse});
  //nofilter_const(Int, C, X_nofilter, Phat);

  //Tensor<double> X_unf("X", {PAO_HAT, AUX, MO}, {Dense, Sparse, Sparse});
  //noncov_mp2_unfused(Int, C, Phat, X_unf, L);
  //Tensor<double> X_nofilter_unf("X", {AUX, MO, PAO_HAT},
  //                              {Dense, Sparse, Sparse});
  //nofilter_unfused(Int, C, X_nofilter_unf, Phat);

  //Tensor<double> X_nary("X", {AUX, MO, PAO_HAT}, {Dense, Sparse, Sparse});
  //noncov_mp2_nary(Int, C, Phat, X_nary, L);
  //Tensor<double> X_nofilter_nary("X", {AUX, MO, PAO_HAT},
  //                               {Dense, Sparse, Sparse});
  //nofilter_nary(Int, C, X_nofilter_nary, Phat);
  return 0;
}
