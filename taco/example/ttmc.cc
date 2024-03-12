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

const int Rank = 50;

void ttmc_fused_innermost(Tensor<double> I, Tensor<double> M2,
                          Tensor<double> R1, Tensor<double> M1) {
  M1 = M1.transpose({1, 0});
  M1.setName("M1");
  IndexVar b("b");
  IndexVar k("k");
  IndexVar j("j");
  IndexVar i("i");
  IndexVar a("a");
  TensorVar I_var = I.getTensorVar();
  TensorVar M2_var = M2.getTensorVar();
  TensorVar R1_var = R1.getTensorVar();
  TensorVar M1_var = M1.getTensorVar();
  Tensor<double> scalar = Tensor<double>("IM2");
  // Tensor<double> scalar("scalar", {0}, {Dense});
  TensorVar IM2 = scalar.getTensorVar();
  // float IM2;
  R1(b, i, a) = I(i, j, k) * M2(b, k) * M1(j, a);
  auto fused_ir = forall(
      b,
      forall(
          i,
          forall(j, where(forall(a, R1_var(b, i, a) += IM2() * M1_var(j, a)),
                          forall(k, IM2() += I_var(i, j, k) * M2_var(b, k))))));
  R1.compile(fused_ir);
  auto start = std::chrono::high_resolution_clock::now();
  R1.assemble();
  R1.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time ttmc1_const:  " << elapsed.count() << " ms "
            << std::endl;
}

void ttmc_nary(Tensor<double> I, Tensor<double> M1, Tensor<double> M2,
               Tensor<double> R, Tensor<double> M3) {
  IndexVar l("l");
  IndexVar k("k");
  IndexVar a("a");
  IndexVar b("b");
  IndexVar j("j");
  IndexVar c("c");
  IndexVar i("i");
  R(i, a, b) += I(i, j, k) * M2(b, k) * M1(a, j);

  R.compile();
  auto start = std::chrono::high_resolution_clock::now();
  R.assemble();
  R.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time ttmc1_TACO-Nary:  " << elapsed.count() << " ms " << std::endl;
}

void ttmc_unfused(Tensor<double> I, Tensor<double> M2, Tensor<double> R,
                  Tensor<double> M1) {
  IndexVar a("a");
  IndexVar k("k");
  IndexVar b("b");
  IndexVar j("j");
  IndexVar i("i");
  TensorVar I_var = I.getTensorVar();
  TensorVar M2_var = M2.getTensorVar();
  TensorVar R_var = R.getTensorVar();
  TensorVar M1_var = M1.getTensorVar();
  Tensor<double> IM2("IM2", {I.getDimension(0), Rank, I.getDimension(1)},
                     {Dense, Dense, Sparse});
  IM2(i, b, j) = I(i, j, k) * M2(b, k);
  R(i, b, a) = IM2(i, b, j) * M1(a, j);
  IM2.compile();
  R.compile();
  auto start = std::chrono::high_resolution_clock::now();
  IM2.assemble();
  IM2.compute();
  R.assemble();
  R.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time ttmc1_TACO-unfused:  " << elapsed.count() << " ms " << std::endl;
}

void generate_ones(Tensor<double> &out) {
  // Tensor<double> t(name, {rows, cols}, {Dense, Dense});
  for (int i = 0; i < out.getDimension(0); i++) {
    for (int j = 0; j < out.getDimension(1); j++) {
      out.insert({i, j}, 1.0);
    }
  }
  out.pack();
}

int main() {

  for (auto file : std::filesystem::directory_iterator(
           std::filesystem::current_path() / "data_frostt/")) {
    std::cout << "****** Running " << file << "********" << std::endl;
    Tensor<double> I1 = read(file.path(), Format({Dense, Sparse, Sparse}));
    I1.pack();
    I1.setName("Input");
    std::cout << "Read I" << std::endl;

    Tensor<double> M1("M1", {Rank, I1.getDimension(1)}, {Dense, Dense});
    generate_ones(M1);

    Tensor<double> M2("M2", {Rank, I1.getDimension(2)}, {Dense, Dense});
    generate_ones(M2);
    Tensor<double> R("result_fused_fullinner", {Rank, I1.getDimension(0), Rank},
                     {Dense, Dense, Dense});
    ttmc_fused_innermost(I1, M2, R, M1);

    Tensor<double> R2("result_unfused", {I1.getDimension(0), Rank, Rank},
                      {Dense, Dense, Dense});
    Tensor<double> i_unfused = I1;
    ttmc_unfused(i_unfused, M2, R2, M1);

    Tensor<double> R3("result_nary", {I1.getDimension(0), Rank, Rank},
                      {Dense, Dense, Dense});
    ttmc_nary(I1, M1, M2, R3, M1);
  }

  return 0;
}
