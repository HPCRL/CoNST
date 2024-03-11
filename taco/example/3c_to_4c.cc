#include "4c_fused.hpp"
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

void nary_3c_to_4c(Tensor<double> I1, Tensor<double> D, Tensor<double> Result,
                   Tensor<double> I2) {
  IndexVar nuhat("nuhat");
  IndexVar i("i");
  IndexVar l("l");
  IndexVar j("j");
  IndexVar k("k");
  IndexVar muhat("muhat");
  I1 = I1.transpose({1, 2, 0});
  I2 = I2.transpose({1, 2, 0});
  Result(i, j, muhat, nuhat) += I1(i, muhat, k) * D(k, l) * I2(j, nuhat, l);
  Result.compile();
  auto start = std::chrono::high_resolution_clock::now();
  Result.assemble();
  Result.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time TACO-Nary_3c_to_4c:  " << elapsed.count() << " ms "
            << std::endl;
  // write("data_3cent_real/4centered_nary.tns", Result);
}

void unfused_3c_to_4c(Tensor<double> I1, Tensor<double> D,
                      Tensor<double> Result, Tensor<double> I2) {
  IndexVar nuhat("nuhat");
  IndexVar i("i");
  IndexVar l("l");
  IndexVar j("j");
  IndexVar k("k");
  IndexVar muhat("muhat");
  I1 = I1.transpose({1, 2, 0});
  I2 = I2.transpose({1, 2, 0});
  Tensor<double> I1D("I1D", {MO, PAO_HAT, AUX},
                     {Dense, Sparse, Sparse}); // I * MU_HAT * L
  I1D(i, muhat, l) = I1(i, muhat, k) * D(l, k);
  Result(i, muhat, j, nuhat) = I1D(i, muhat, l) * I2(j, nuhat, l);
  I1D.compile();
  Result.compile();
  auto start = std::chrono::high_resolution_clock::now();
  I1D.assemble();
  I1D.compute();
  Result.assemble();
  Result.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time TACO-unfused_3c_to_4c:  " << elapsed.count() << " ms "
            << std::endl;
  // write("data_3cent_real/4centered_unfused.tns", Result);
}

int main(int argc, char **argv) {
  assert(argc == 2);
  std::string exp_size = argv[1];
  std::string path_prefix = parseExperiment(exp_size);
  Tensor<double> I1("Int", {AUX, MO, PAO_HAT}, {Dense, Sparse, Sparse});
  I1 = read(path_prefix + "/threec_int.tns", {Dense, Sparse, Sparse});
  I1.pack();
  // I1 = I1.transpose({1, 2, 0});
  I1.setName("I1");

  Tensor<double> I2 = I1;

  Tensor<double> D("D", {AUX, AUX}, {Dense, Dense});
  D = read(path_prefix + "/D.tns", {Dense, Dense});
  D.pack();
  D.setName("D");

  Tensor<double> Res("Res", {PAO_HAT, MO, PAO_HAT, MO},
                     {Dense, Sparse, Sparse, Sparse});
  const_3c_to_4c(I1, D, Res, I2);

  Tensor<double> Res_unf("Res_unf", {MO, PAO_HAT, MO, PAO_HAT},
                         {Dense, Sparse, Sparse, Sparse});
  unfused_3c_to_4c(I1, D, Res_unf, I2);

  Tensor<double> Res_nary("Res_nary", {MO, MO, PAO_HAT, PAO_HAT},
                          {Dense, Sparse, Sparse, Sparse});
  nary_3c_to_4c(I1, D, Res_nary, I2);

  return 0;
}
