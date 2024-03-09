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

void ttmc_fused2(Tensor<double> I, Tensor<double> M2, Tensor<double> R2,
                 Tensor<double> M1) {
  IndexVar k("k");
  IndexVar a("a");
  IndexVar j("j");
  IndexVar b("b");
  IndexVar i("i");
  TensorVar I_var = I.getTensorVar();
  TensorVar M2_var = M2.getTensorVar();
  TensorVar R2_var = R2.getTensorVar();
  TensorVar M1_var = M1.getTensorVar();
  TensorVar IM2 = TensorVar(
      "IM2", Type(I.getComponentType(), {I.getDimension(2)}), {taco::dense});
  R2(j, b, a) = I(j, k, i) * M2(b, k) * M1(a, i);
  auto fused_ir =
      forall(j, forall(b, where(forall(a, forall(i, R2_var(j, b, a) +=
                                                    IM2(i) * M1_var(a, i))),
                                forall(k, forall(i, IM2(i) += I_var(j, k, i) *
                                                              M2_var(b, k))))));
  R2.compile(fused_ir);
  auto start = std::chrono::high_resolution_clock::now();
  R2.assemble();
  R2.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time ttmc_fused2:  " << elapsed.count() << " ms " << std::endl;
}

void ttmc_fused2_innermost(Tensor<double> I, Tensor<double> M2,
                           Tensor<double> R2, Tensor<double> M1) {
  M1 = M1.transpose({1, 0});
  M1.setName("M1");
  IndexVar b("b");
  IndexVar a("a");
  IndexVar j("j");
  IndexVar k("k");
  IndexVar i("i");
  TensorVar I_var = I.getTensorVar();
  TensorVar M2_var = M2.getTensorVar();
  TensorVar R2_var = R2.getTensorVar();
  TensorVar M1_var = M1.getTensorVar();
  Tensor<double> scalar = Tensor<double>("IM2");
  TensorVar IM2 = scalar.getTensorVar();
  R2(b, j, a) = I(j, i, k) * M2(b, k) * M1(i, a);
  auto fused_ir = forall(
      b,
      forall(
          j,
          forall(i, where(forall(a, R2_var(b, j, a) += IM2() * M1_var(i, a)),
                          forall(k, IM2() += I_var(j, i, k) * M2_var(b, k))))));
  ;
  R2.compile(fused_ir);
  auto start = std::chrono::high_resolution_clock::now();
  R2.assemble();
  R2.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time ttmc_fused2_innermost:  " << elapsed.count() << " ms "
            << std::endl;
}

void ttmc_nary(Tensor<double> I, Tensor<double> M1, Tensor<double> M2,
               Tensor<double> R) {
  IndexVar l("l");
  IndexVar k("k");
  IndexVar a("a");
  IndexVar b("b");
  IndexVar j("j");
  IndexVar c("c");
  IndexVar i("i");
  R(j, b, a) += I(j, k, i) * M2(b, k) * M1(a, i);

  R.compile();
  auto start = std::chrono::high_resolution_clock::now();
  R.assemble();
  R.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time ttmc_nary2:  " << elapsed.count() << " ms " << std::endl;
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
  IM2(j, b, i) = I(j, i, k) * M2(b, k);
  R(j, b, a) = IM2(j, b, i) * M1(a, i);
  IM2.compile();
  R.compile();
  auto start = std::chrono::high_resolution_clock::now();
  IM2.assemble();
  IM2.compute();
  R.assemble();
  R.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time ttmc_unfused2:  " << elapsed.count() << " ms "
            << std::endl;
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
    // Tensor<double> I1 = read("inp_scipy.tns", {Dense, Sparse, Sparse});
    I1.pack();
    I1.setName("Input");
    std::cout << "Read I" << std::endl;

    Tensor<double> M1("M1", {Rank, I1.getDimension(0)}, {Dense, Dense});
    generate_ones(M1);

    Tensor<double> M2("M2", {Rank, I1.getDimension(2)}, {Dense, Dense});
    generate_ones(M2);

    // Tensor<double> R("result_fused", {I1.getDimension(1), Rank, Rank},
    //                  {Dense, Dense, Dense});
    // Tensor<double> i_fused = I1.transpose({1, 2, 0});
    // ttmc_fused2(i_fused, M2, R, M1);

    //Tensor<double> R1("result_unfused", {Rank, I1.getDimension(1), Rank},
    //                  {Dense, Dense, Dense});
    //Tensor<double> i_fused_innermost = I1.transpose({1, 0, 2});
    //ttmc_fused2_innermost(i_fused_innermost, M2, R1, M1);

    Tensor<double> R2("result_unfused", {I1.getDimension(1), Rank, Rank},
                      {Dense, Dense, Dense});
    Tensor<double> i_unfused = I1.transpose({1, 0, 2});
    ttmc_unfused(i_unfused, M2, R2, M1);

    //Tensor<double> R3("result_nary", {I1.getDimension(1), Rank, Rank},
    //                  {Dense, Dense, Dense});
    //Tensor<double> i_nary = I1.transpose({1, 2, 0});
    //ttmc_nary(i_nary, M1, M2, R3);
  }

  return 0;
}
