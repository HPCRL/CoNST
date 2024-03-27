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
void mttkrp_fused3(Tensor<double> I, Tensor<double> M2, Tensor<double> R,
                   Tensor<double> M1) {
  IndexVar k("k");
  IndexVar i("i");
  IndexVar r("r");
  IndexVar j("j");
  TensorVar I_var = I.getTensorVar();
  TensorVar M2_var = M2.getTensorVar();
  TensorVar R_var = R.getTensorVar();
  TensorVar M1_var = M1.getTensorVar();
  TensorVar IM2 =
      TensorVar("IM2", Type(I.getComponentType(), {Rank}), {taco::dense});
  R(k, r) = I(k, j, i) * M2(i, r) * M1(j, r);
  auto fused_ir =
      forall(k, forall(j, where(forall(r, R_var(k, r) += IM2(r) * M1_var(j, r)),
                                forall(i, forall(r, IM2(r) += I_var(k, j, i) *
                                                              M2_var(i, r))))));
  R.compile(fused_ir);
  auto start = std::chrono::high_resolution_clock::now();
  R.assemble();
  R.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time mttkrp3_const:  " << elapsed.count() << " ms "
            << std::endl;
}

void mttkrp_nary3(Tensor<double> I, Tensor<double> M2, Tensor<double> R,
                  Tensor<double> M1) {
  IndexVar k("k");
  IndexVar i("i");
  IndexVar r("r");
  IndexVar j("j");
  R(k, r) = I(k, j, i) * M2(i, r) * M1(j, r);
  R.compile();
  auto start = std::chrono::high_resolution_clock::now();
  R.assemble();
  R.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time mttkrp3_TACO-Nary:  " << elapsed.count() << " ms "
            << std::endl;
}

void mttkrp_unfused3(Tensor<double> I, Tensor<double> M2, Tensor<double> R,
                     Tensor<double> M1) {
  IndexVar k("k");
  IndexVar i("i");
  IndexVar r("r");
  IndexVar j("j");
  M2 = M2.transpose({1, 0});
  M2.setName("M2");
  M1 = M1.transpose({1, 0});
  M1.setName("M1");
  Tensor<double> IM2("IM2", {Rank, I.getDimension(0), I.getDimension(1)},
                     {Dense, Sparse, Sparse});
  IM2(r, k, j) += I(k, j, i) * M2(r, i);
  R(r, k) = IM2(r, k, j) * M1(r, j);
  IM2.compile();
  R.compile();
  auto start = std::chrono::high_resolution_clock::now();
  IM2.assemble();
  R.assemble();
  IM2.compute();
  R.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time mttkrp3_TACO-unfused:  " << elapsed.count() << " ms "
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
  // Tensor<double> I1 = read("inp_scipy.tns", {Dense, Sparse, Sparse});
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
  Tensor<double> inp = I1.transpose({2, 1, 0});

  Tensor<double> M1("M1", {I1.getDimension(1), Rank}, {Dense, Dense});
  generate_ones(M1);

  Tensor<double> M2("M2", {I1.getDimension(0), Rank}, {Dense, Dense});
  generate_ones(M2);

  if (method_index == 0) {
    Tensor<double> R("result_fused", {I1.getDimension(2), Rank},
                     {Dense, Dense});
    mttkrp_fused3(inp, M2, R, M1);
  } else if (method_index == 1) {

    Tensor<double> R2("result_unfused", {Rank, I1.getDimension(2)},
                      {Dense, Dense});
    // Tensor<double> inp_unfused = I1.transpose({2, 1, 0});
    mttkrp_unfused3(inp, M2, R2, M1);
  } else if (method_index == 2) {

    Tensor<double> R3("result_nary", {I1.getDimension(2), Rank},
                      {Dense, Dense});
    // Tensor<double> inp_fused = I1.transpose({2, 1, 0});
    mttkrp_nary3(inp, M2, R3, M1);
  }

  return 0;
}
