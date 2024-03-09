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

// const int II = 600;
// const int J = 600;
// const int K = 600;
// const int A = 50;
// const int B = 50;
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
  std::cout << "Time ttmc_fused_innermost:  " << elapsed.count() << " ms "
            << std::endl;
}

void ttmc_fused(Tensor<double> I, Tensor<double> M2, Tensor<double> R,
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
  TensorVar IM2 = TensorVar(
      "IM2", Type(I.getComponentType(), {I.getDimension(2)}), {taco::dense});
  R(b, i, a) = I(i, k, j) * M2(b, k) * M1(a, j);
  auto fused_ir =
      forall(b, forall(i, where(forall(a, forall(j, R_var(b, i, a) +=
                                                    IM2(j) * M1_var(a, j))),
                                forall(k, forall(j, IM2(j) += I_var(i, k, j) *
                                                              M2_var(b, k))))));
  R.compile(fused_ir);
  auto start = std::chrono::high_resolution_clock::now();
  R.assemble();
  R.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time ttmc_fused:  " << elapsed.count() << " ms " << std::endl;
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
  std::cout << "Time ttmc_nary:  " << elapsed.count() << " ms " << std::endl;
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
  std::cout << "Time ttmc_unfused:  " << elapsed.count() << " ms " << std::endl;
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
  // Tensor<double> I1("input", {II, J, K}, {Dense, Sparse, Sparse});

  for (auto file : std::filesystem::directory_iterator(
           std::filesystem::current_path() / "data_frostt/")) {
    std::cout << "****** Running " << file << "********" << std::endl;
    Tensor<double> I1 = read(file.path(), Format({Dense, Sparse, Sparse}));
    //   delete &raw_tensor;
    // Tensor<double> I1 = read("inp_scipy.tns", {Dense, Sparse, Sparse});
    I1.pack();
    I1.setName("Input");
    std::cout << "Read I" << std::endl;

    Tensor<double> M1("M1", {Rank, I1.getDimension(1)}, {Dense, Dense});
    generate_ones(M1);
    // Tensor<double> M1("M1", {J, A}, {Dense, Dense});
    // M1 = read("m1.tns", {Dense, Dense});
    // M1 = M1.transpose({1, 0});
    // M1.setName("M1");
    // M1.pack();
    // std::cout << "Read M1" << std::endl;

    Tensor<double> M2("M2", {Rank, I1.getDimension(2)}, {Dense, Dense});
    generate_ones(M2);
    // Tensor<double> M2("M2", {K, B}, {Dense, Dense});
    // M2 = read("m2.tns", {Dense, Dense});
    // M2 = M2.transpose({1, 0});
    // M2.setName("M2");
    // M2.pack();
    // std::cout << "Read M2" << std::endl;
    //
    //Tensor<double> R("result_fused_fullinner", {Rank, I1.getDimension(0), Rank},
    //                 {Dense, Dense, Dense});
    //Tensor<double> i_fused = I1.transpose({0, 1, 2});
    //ttmc_fused_innermost(i_fused, M2, R, M1);

    //Tensor<double> R1("result_fused", {Rank, I1.getDimension(0), Rank},
    //                  {Dense, Dense, Dense});
    //Tensor<double> i_fused2 = I1.transpose({0, 2, 1});
    //ttmc_fused(i_fused2, M2, R1, M1);

    Tensor<double> R2("result_unfused", {I1.getDimension(0), Rank, Rank},
                      {Dense, Dense, Dense});
    Tensor<double> i_unfused = I1;
    ttmc_unfused(i_unfused, M2, R2, M1);

    //Tensor<double> R3("result_nary", {I1.getDimension(0), Rank, Rank},
    //                  {Dense, Dense, Dense});
    //ttmc_nary(I1, M1, M2, R3, M1);
  }

  return 0;
}
