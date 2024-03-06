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

const int MO = 10;
const int PAO = 20;
const int AUX = 50;
const int PNO = 30;

IndexVar e_mu("e_mu");
IndexVar i("i");
IndexVar m("m");
IndexVar n("n");
IndexVar j("j");
IndexVar k("k");
IndexVar e_j("e_j");

void unfused_int4c(Tensor<double> teoo, Tensor<double> d, Tensor<double> Result,
                   Tensor<double> teov) {
  Tensor<double> pao_cont("pao_cont", {MO, MO, AUX, PNO},
                          {Dense, Sparse, Dense, Dense}); // I * MU_HAT * L
  pao_cont(n, j, k, e_j) = d(j, e_mu, e_j) * teov(n, e_mu, k);
  Result(m, i, n, j, e_j) = teoo(m, i, k) * pao_cont(n, j, k, e_j);
  pao_cont.compile();
  Result.compile();
  auto start = std::chrono::high_resolution_clock::now();
  pao_cont.assemble();
  pao_cont.compute();
  Result.assemble();
  Result.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time unfused_int4c:  " << elapsed.count() << " ms "
            << std::endl;
  write("res_unfused.tns", Result);
}

void nary_int4c(Tensor<double> teoo, Tensor<double> d, Tensor<double> Result,
                Tensor<double> teov) {
  Result(m, i, n, j, e_j) = teoo(m, i, k) * d(j, e_mu, e_j) * teov(n, e_mu, k);
  Result.compile();
  auto start = std::chrono::high_resolution_clock::now();
  Result.assemble();
  Result.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time int4c_nary:  " << elapsed.count() << " ms " << std::endl;
  write("res_nary.tns", Result);
}

int main() {
  Tensor<double> teoo("teoo", {MO, MO, AUX}, {Dense, Sparse, Sparse});
  teoo = read("teoo.tns", {Dense, Sparse, Sparse});
  teoo.pack();
  Tensor<double> teov("teov", {MO, PAO, AUX}, {Dense, Sparse, Sparse});
  teov = read("teov.tns", {Dense, Sparse, Sparse});
  teov.pack();
  Tensor<double> d("d", {MO, PAO, PNO}, {Dense, Sparse, Dense});
  d = read("d.tns", {Dense, Sparse, Dense});
  d.pack();
  Tensor<double> res("Result", {MO, MO, MO, MO, PNO},
                     {Dense, Sparse, Sparse, Sparse, Dense});

  unfused_int4c(teoo, d, res, teov);
  Tensor<double> res_nary("Result", {MO, MO, MO, MO, PNO},
                     {Dense, Sparse, Sparse, Sparse, Dense});
  nary_int4c(teoo, d, res_nary, teov);

  return 0;
}
