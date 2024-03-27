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

void mttkrp_fused1(Tensor<double> I, Tensor<double> M1, Tensor<double> R,
                   Tensor<double> M2) {
  IndexVar k("k");
  IndexVar i("i");
  IndexVar r("r");
  IndexVar j("j");
  M1 = M1.transpose({1, 0});
  M1.setName("M1");
  M2 = M2.transpose({1, 0});
  M2.setName("M2");
  TensorVar I_var = I.getTensorVar();
  TensorVar M1_var = M1.getTensorVar();
  TensorVar R_var = R.getTensorVar();
  TensorVar M2_var = M2.getTensorVar();
  TensorVar IM1 =
      TensorVar("IM1", Type(I.getComponentType(), {Rank}), {taco::dense});
  R(i, r) = I(k, i, j) * M1(j, r) * M2(k, r);
  auto fused_ir =
      forall(k, forall(i, where(forall(r, R_var(i, r) += IM1(r) * M2_var(k, r)),
                                forall(j, forall(r, IM1(r) += I_var(k, i, j) *
                                                              M1_var(j, r))))));
  R.compile(fused_ir);
  auto start = std::chrono::high_resolution_clock::now();
  R.assemble();
  R.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time mttkrp1_const:  " << elapsed.count() << " ms "
            << std::endl;
  // write("nell1-1_fused.tns", R);
}

void mttkrp_nary(Tensor<double> I, Tensor<double> M1, Tensor<double> M2,
                 Tensor<double> R) {
  IndexVar r("r");
  IndexVar i("i");
  IndexVar k("k");
  IndexVar j("j");
  R(i, r) += I(i, j, k) * M2(r, k) * M1(r, j);
  R.compile();
  auto start = std::chrono::high_resolution_clock::now();
  R.assemble();
  R.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time mttkrp1_TACO-Nary:  " << elapsed.count() << " ms "
            << std::endl;
  // write("nell1-1_nary.tns", R);
}

void mttkrp_unfused(Tensor<double> I, Tensor<double> M2, Tensor<double> R,
                    Tensor<double> M1) {
  IndexVar r("r");
  IndexVar i("i");
  IndexVar k("k");
  IndexVar j("j");
  Tensor<double> IM2("IM2", {Rank, I.getDimension(0), I.getDimension(1)},
                     {Dense, Sparse, Sparse});
  IM2(r, i, j) = I(i, j, k) * M2(r, k);
  R(r, i) = IM2(r, i, j) * M1(r, j);
  IM2.compile();
  R.compile();
  auto start = std::chrono::high_resolution_clock::now();
  IM2.assemble();
  IM2.compute();
  R.assemble();
  R.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time mttkrp1_TACO-unfused:  " << elapsed.count() << " ms "
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

int main(int argc, char *argv[]) {
  assert(argc == 3);
  std::string tensor_name = argv[1];
  int method_index = std::stoi(argv[2]);
  if (method_index == 0) {
    std::cout << "****** Running " << tensor_name << "********" << std::endl;
  }
  auto fpath = std::filesystem::current_path() / "data_frostt/" / tensor_name;
  Tensor<double> I1 = read(fpath.string(), Format({Dense, Sparse, Sparse}));
  // Tensor<double> I1 = read("inp_scipy.tns", {Dense, Sparse, Sparse});
  I1.pack();
  I1.setName("Input");
  std::cout << "Read I" << std::endl;

  Tensor<double> M1("M1", {Rank, I1.getDimension(1)}, {Dense, Dense});
  generate_ones(M1);

  Tensor<double> M2("M2", {Rank, I1.getDimension(2)}, {Dense, Dense});
  generate_ones(M2);

  if (method_index == 0) {
    Tensor<double> R1("result_fused", {I1.getDimension(0), Rank},
                      {Dense, Dense});
    Tensor<double> inp = I1.transpose({2, 0, 1});
    mttkrp_fused1(inp, M1, R1, M2);
  } else if (method_index == 1) {

    Tensor<double> R3("result_nary", {I1.getDimension(0), Rank},
                      {Dense, Dense});
    mttkrp_nary(I1, M1, M2, R3);
  } else if (method_index == 2) {
    Tensor<double> R2("result_unfused", {Rank, I1.getDimension(0)},
                      {Dense, Dense});
    mttkrp_unfused(I1, M2, R2, M1);
  }

  return 0;
}
