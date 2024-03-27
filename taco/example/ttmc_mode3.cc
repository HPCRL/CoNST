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

void ttmc_fused3_innermost(Tensor<double> I, Tensor<double> M2,
                           Tensor<double> R3, Tensor<double> M1) {
  M1 = M1.transpose({1, 0});
  M1.setName("M1");
  IndexVar a("a");
  IndexVar b("b");
  IndexVar k("k");
  IndexVar j("j");
  IndexVar i("i");
  TensorVar I_var = I.getTensorVar();
  TensorVar M2_var = M2.getTensorVar();
  TensorVar R3_var = R3.getTensorVar();
  TensorVar M1_var = M1.getTensorVar();

  Tensor<double> scalar = Tensor<double>("IM2");
  TensorVar IM2 = scalar.getTensorVar();
  R3(b, k, a) = I(i, k, j) * M2(b, j) * M1(i, a);
  auto fused_ir = forall(
      i,
      forall(
          b,
          forall(k, where(forall(a, R3_var(b, k, a) += IM2() * M1_var(i, a)),
                          forall(j, IM2() += I_var(i, k, j) * M2_var(b, j))))));
  R3.compile(fused_ir);
  auto start = std::chrono::high_resolution_clock::now();
  R3.assemble();
  R3.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time ttmc3_const:  " << elapsed.count() << " ms " << std::endl;
}

void ttmc_nary3(Tensor<double> I, Tensor<double> M1, Tensor<double> M2,
                Tensor<double> R) {
  IndexVar l("l");
  IndexVar k("k");
  IndexVar a("a");
  IndexVar b("b");
  IndexVar j("j");
  IndexVar c("c");
  IndexVar i("i");
  R(k, b, a) += I(k, j, i) * M2(b, j) * M1(a, i);

  R.compile();
  auto start = std::chrono::high_resolution_clock::now();
  R.assemble();
  R.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time ttmc3_TACO-Nary:  " << elapsed.count() << " ms "
            << std::endl;
}

void ttmc_unfused3(Tensor<double> I, Tensor<double> M2, Tensor<double> R,
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
  Tensor<double> IM2("IM2", {Rank, I.getDimension(0), I.getDimension(1)},
                     {Dense, Sparse, Sparse});
  IM2(b, k, i) = I(k, i, j) * M2(b, j);
  R(b, k, a) = IM2(b, k, i) * M1(a, i);
  IM2.compile();
  R.compile();
  auto start = std::chrono::high_resolution_clock::now();
  IM2.assemble();
  IM2.compute();
  R.assemble();
  R.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time ttmc3_TACO-unfused:  " << elapsed.count() << " ms "
            << std::endl;
}

void generate_ones(Tensor<double> &out) {
  for (int i = 0; i < out.getDimension(0); i++) {
    for (int j = 0; j < out.getDimension(1); j++) {
      out.insert({i, j}, 1.0);
    }
  }
  out.pack();
}

int main(int argc, char *argv[]) {

  assert(argc == 3);
  std::string tensor_name = argv[1];
  int method_index = std::stoi(argv[2]);
  if (method_index == 0) {
    std::cout << "****** Running " << tensor_name << "********" << std::endl;
  }
  auto fpath = std::filesystem::current_path() / "data_frostt/" / tensor_name;
  Tensor<double> I1 = read(fpath.string(), Format({Dense, Sparse, Sparse}));

  I1.pack();
  I1.setName("Input");
  std::cout << "Read I" << std::endl;

  Tensor<double> M1("M1", {Rank, I1.getDimension(0)}, {Dense, Dense});
  generate_ones(M1);

  Tensor<double> M2("M2", {Rank, I1.getDimension(1)}, {Dense, Dense});
  generate_ones(M2);

  if (method_index == 0) {
    Tensor<double> R1("result_fusedinnermost", {Rank, I1.getDimension(2), Rank},
                      {Dense, Dense, Dense});
    Tensor<double> i_fused_innermost = I1.transpose({0, 2, 1});
    ttmc_fused3_innermost(i_fused_innermost, M2, R1, M1);
  } else if (method_index == 1) {

    Tensor<double> R2("result_unfused", {Rank, I1.getDimension(2), Rank},
                      {Dense, Dense, Dense});
    Tensor<double> i_unfused = I1.transpose({2, 0, 1});
    ttmc_unfused3(i_unfused, M2, R2, M1);
  } else if (method_index == 2) {

    Tensor<double> R3("result_nary", {I1.getDimension(2), Rank, Rank},
                      {Dense, Dense, Dense});
    Tensor<double> i_nary = I1.transpose({2, 1, 0});
    ttmc_nary3(i_nary, M1, M2, R3);
  }

  return 0;
}
