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

const int MO = 100;
const int PAO_HAT = 300;
const int AUX = 700;

void fused_3c_to_4c(Tensor<double> I1, Tensor<double> D, Tensor<double> Result,
                    Tensor<double> I2) {
  IndexVar nuhat("nuhat");
  IndexVar i("i");
  IndexVar l("l");
  IndexVar j("j");
  IndexVar k("k");
  IndexVar muhat("muhat");
  TensorVar I1_var = I1.getTensorVar();
  TensorVar D_var = D.getTensorVar();
  TensorVar Result_var = Result.getTensorVar();
  TensorVar I2_var = I2.getTensorVar();
  TensorVar I1D =
      TensorVar("I1D", Type(I1.getComponentType(), {AUX}), {taco::dense});
  Result(i, muhat, j, nuhat) = I1(i, muhat, k) * D(k, l) * I2(j, nuhat, l);
  auto fused_ir = forall(
      i,
      forall(muhat,
             where(forall(j, forall(nuhat,
                                    forall(l, Result_var(i, muhat, j, nuhat) +=
                                              I1D(l) * I2_var(j, nuhat, l)))),
                   forall(k, forall(l, I1D(l) +=
                                       I1_var(i, muhat, k) * D_var(k, l))))));
  Result.compile(fused_ir);
  auto start = std::chrono::high_resolution_clock::now();
  Result.assemble();
  Result.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time 3c_to_4c_fused:  " << elapsed.count() << " ms "
            << std::endl;
  // write("data_3cent_real/4centered_fused.tns", Result);
}
//
// void nary_3c_to_4c(Tensor<double> I1, Tensor<double> D, Tensor<double>
// Result,
//                   Tensor<double> I2) {
//  IndexVar nuhat("nuhat");
//  IndexVar i("i");
//  IndexVar l("l");
//  IndexVar j("j");
//  IndexVar k("k");
//  IndexVar muhat("muhat");
//  Result(i, j, muhat, nuhat) += I1(i, muhat, k) * D(k, l) * I2(j, nuhat, l);
//  Result.compile();
//  auto start = std::chrono::high_resolution_clock::now();
//  Result.assemble();
//  Result.compute();
//  auto end = std::chrono::high_resolution_clock::now();
//  std::chrono::duration<double, std::milli> elapsed = end - start;
//  std::cout << "Time 3c_to_4c_nary:  " << elapsed.count() << " ms "
//            << std::endl;
//  Result = Result.transpose({0, 2, 1, 3});
//  //write("data_3cent_real/4centered_nary.tns", Result);
//}

void unfused_3c_to_4c(Tensor<double> I1, Tensor<double> D,
                      Tensor<double> Result, Tensor<double> I2) {
  IndexVar nuhat("nuhat");
  IndexVar i("i");
  IndexVar l("l");
  IndexVar j("j");
  IndexVar k("k");
  IndexVar muhat("muhat");
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
  std::cout << "Time 3c_to_4c_unfused:  " << elapsed.count() << " ms "
            << std::endl;
  // write("data_3cent_real/4centered_unfused.tns", Result);
}

int main() {
  Tensor<double> I1("Int", {AUX, MO, PAO_HAT}, {Dense, Sparse, Sparse});
  I1 = read("data_3cent_realmid/threec_int.tns", {Dense, Sparse, Sparse});
  I1.pack();
  I1 = I1.transpose({1, 2, 0});
  I1.setName("I1");

  Tensor<double> I2 = I1;

  Tensor<double> D("D", {AUX, AUX}, {Dense, Dense});
  D = read("data_3cent_realmid/D.tns", {Dense, Dense});
  D.pack();
  D.setName("D");

  Tensor<double> Res("Res", {MO, PAO_HAT, MO, PAO_HAT},
                     {Dense, Sparse, Sparse, Sparse});
  fused_3c_to_4c(I1, D, Res, I2);

  // Tensor<double> Res_nary("Res_nary", {MO, MO, PAO_HAT, PAO_HAT},
  //                    {Dense, Sparse, Sparse, Sparse});
  // nary_3c_to_4c(I1, D, Res_nary, I2);

  //Tensor<double> Res_unf("Res_unf", {MO, PAO_HAT, MO, PAO_HAT},
  //                   {Dense, Sparse, Sparse, Sparse});
  //unfused_3c_to_4c(I1, D, Res_unf, I2);

  return 0;
}
