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

// const int II = 500;
// const int J = 500;
// const int K = 500;
const int Rank = 50;

void mttkrp_fused2(Tensor<double> I, Tensor<double> M2, Tensor<double> R,
                   Tensor<double> M1) {
  IndexVar k("k");
  IndexVar i("i");
  IndexVar r("r");
  IndexVar j("j");
  M1 = M1.transpose({1, 0});
  M1.setName("M1");
  M2 = M2.transpose({1, 0});
  M2.setName("M2");
  TensorVar I_var = I.getTensorVar();
  TensorVar M2_var = M2.getTensorVar();
  TensorVar R_var = R.getTensorVar();
  TensorVar M1_var = M1.getTensorVar();
  TensorVar IM2 =
      TensorVar("IM2", Type(I.getComponentType(), {Rank}), {taco::dense});
  R(j, r) = I(j, i, k) * M2(k, r) * M1(i, r);
  auto fused_ir =
      forall(j, forall(i, where(forall(r, R_var(j, r) += IM2(r) * M1_var(i, r)),
                                forall(k, forall(r, IM2(r) += I_var(j, i, k) *
                                                              M2_var(k, r))))));
  R.compile(fused_ir);
  auto start = std::chrono::high_resolution_clock::now();
  R.assemble();
  R.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time mttkrp2_const:  " << elapsed.count() << " ms "
            << std::endl;
}

void mttkrp_nary(Tensor<double> I, Tensor<double> M1, Tensor<double> M2,
                 Tensor<double> R) {
  IndexVar r("r");
  IndexVar i("i");
  IndexVar k("k");
  IndexVar j("j");
  R(j, r) += I(j, i, k) * M2(r, k) * M1(r, i);
  R.compile();
  auto start = std::chrono::high_resolution_clock::now();
  R.assemble();
  R.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time mttkrp2_TACO-Nary:  " << elapsed.count() << " ms " << std::endl;
}

void mttkrp_unfused(Tensor<double> I, Tensor<double> M2, Tensor<double> R,
                    Tensor<double> M1) {
  IndexVar r("r");
  IndexVar i("i");
  IndexVar k("k");
  IndexVar j("j");
  Tensor<double> IM2("IM2", {Rank, I.getDimension(0), I.getDimension(1)},
                     {Dense, Sparse, Sparse});
  IM2(r, j, i) = I(j, i, k) * M2(r, k);
  R(r, j) = IM2(r, j, i) * M1(r, i);
  IM2.compile();
  R.compile();
  auto start = std::chrono::high_resolution_clock::now();
  IM2.assemble();
  IM2.compute();
  R.assemble();
  R.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time mttkrp2_TACO-unfused:  " << elapsed.count() << " ms "
            << std::endl;
  R = R.transpose({1, 0});
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

int main(int argc, char *argv[]) {
  // std::string path = argv[1];
  for (auto file : std::filesystem::directory_iterator(
           std::filesystem::current_path() / "data_frostt/")) {
    std::cout << "****** Running " << file << "********" << std::endl;
    Tensor<double> I1 = read(file.path(), Format({Dense, Sparse, Sparse}));
    // Tensor<double> I1 = read("inp_scipy.tns", {Dense, Sparse, Sparse});
    I1.pack();
    I1 = I1.transpose({1, 0, 2});
    I1.setName("Input");
    std::cout << "Read I" << std::endl;

    Tensor<double> M1("M1", {Rank, I1.getDimension(1)}, {Dense, Dense});
    generate_ones(M1);

    Tensor<double> M2("M2", {Rank, I1.getDimension(2)}, {Dense, Dense});
    generate_ones(M2);

    Tensor<double> R1("result_fused", {I1.getDimension(0), Rank},
                      {Dense, Dense});
    mttkrp_fused2(I1, M2, R1, M1);
    Tensor<double> R2("result_unfused", {Rank, I1.getDimension(0)},
                      {Dense, Dense});
    mttkrp_unfused(I1, M2, R2, M1);
    Tensor<double> R3("result_nary", {I1.getDimension(0), Rank},
                      {Dense, Dense});
    mttkrp_nary(I1, M1, M2, R3);
  }

  return 0;
}
