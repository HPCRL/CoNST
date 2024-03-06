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
const int I = 100;
const int J = 100;
const int K = 100;
const int L = 100;

IndexVar i("i"), j("j"), k("k"), l("l");
IndexVar ivars[4] = {i, j, k, l};

std::vector<IndexVar> get_in_order(std::set<IndexVar> &required_vars,
                                   IndexVar sorted_vars[4]) {
  std::vector<IndexVar> in_order;
  for (int i = 0; i < 4; i++) {
    auto ivar = sorted_vars[i];
    if (required_vars.find(ivar) != required_vars.end())
      in_order.push_back(ivar);
  }
  return in_order;
}

std::vector<int> get_transpose_indices(std::vector<IndexVar> &in_order,
                                       std::vector<IndexVar> required_vars) {
  std::vector<int> transpose_indices;
  for (size_t i = 0; i < in_order.size(); i++) {
    auto ivar = in_order[i];
    auto it = std::find(required_vars.begin(), required_vars.end(), ivar);
    transpose_indices.push_back(it - required_vars.begin());
  }
  return transpose_indices;
}

Tensor<double> run_contraction(Tensor<double> X, Tensor<double> Y,
                               int ivar_permutation[4]) {

  // Implement:
  // R[i, j, l] += X[i, j, k] * Y[l, k]
  Tensor<double> res("res", {I, J, L}, {Dense, Sparse, Sparse});
  std::set<IndexVar> x_vars = {i, j, k};
  std::set<IndexVar> y_vars = {l, k};
  std::set<IndexVar> res_vars = {i, j, l};

  IndexVar sorted_vars[4];
  for (int i = 0; i < 4; i++) {
    sorted_vars[ivar_permutation[i]] = ivars[i];
  }
  std::cout << "Sorted variables are ";
  for (int i = 0; i < 4; i++) {
    std::cout << sorted_vars[i] << " ";
  }
  std::cout << std::endl;
  // First get the expression as per given random permuatation
  std::vector<IndexVar> x_sorted_vars = get_in_order(x_vars, sorted_vars);
  std::vector<IndexVar> y_sorted_vars = get_in_order(y_vars, sorted_vars);
  std::vector<IndexVar> res_sorted_vars = get_in_order(res_vars, sorted_vars);

  // Now figure out how to transpose the tensor to match this permutation
  std::vector<int> x_transpose_indices =
      get_transpose_indices(x_sorted_vars, std::vector<IndexVar>({i, j, k}));
  std::vector<int> y_transpose_indices =
      get_transpose_indices(y_sorted_vars, std::vector<IndexVar>({l, k}));
  // Now do the transpose
  X = X.transpose(x_transpose_indices);
  X.setName("X");
  Y = Y.transpose(y_transpose_indices);
  Y.setName("Y");

  std::vector<int> res_transpose_indices =
      get_transpose_indices(res_sorted_vars, std::vector<IndexVar>({i, j, l}));

  std::cout << "Expression 1: res[";
  for (auto i : res_sorted_vars) {
    std::cout << i << ", ";
  }
  std::cout << "] = ";
  std::cout << "x[";
  for (auto i : x_sorted_vars) {
    std::cout << i << ", ";
  }
  std::cout << "] * y[";
  for (auto i : y_sorted_vars) {
    std::cout << i << ", ";
  }
  std::cout << "]" << std::endl;

  res(res_sorted_vars[0], res_sorted_vars[1], res_sorted_vars[2]) +=
      X(x_sorted_vars[0], x_sorted_vars[1], x_sorted_vars[2]) *
      Y(y_sorted_vars[0], y_sorted_vars[1]);

  res.compile();
  res.assemble();
  res.compute();
  res = res.transpose(res_transpose_indices);
  res.setName("res");
  return res;
}

Tensor<double> run_fullinner(Tensor<double> X, Tensor<double> Y) {
  Tensor<double> res("res", {I, J, L}, {Dense, Sparse, Sparse});
  res(i, j, l) += X(i, j, k) * Y(l, k);
  res.compile();
  res.assemble();
  res.compute();
  return res;
}

int main(int argc, char *argv[]) {
  Tensor<double> x("x", {I, J, K}, {Dense, Sparse, Sparse});
  x = read("x.tns", {Dense, Sparse, Sparse});
  x.pack();
  Tensor<double> y("y", {L, K}, {Dense, Sparse});
  y = read("y.tns", {Dense, Sparse});
  y.pack();
  Tensor<double> ground_truth = run_fullinner(x, y);
  //int permutation[4] = {0, 1, 2, 3};
  int permutation[4] = {0, 2, 1, 3};
  try {
    auto this_result = run_contraction(x, y, permutation);
    if (equals(this_result, ground_truth)) {
      std::cout << "Correctness check passed!" << std::endl;
    } else {
      std::cout << "Correctness check failed!" << std::endl;
    }
  } catch (TacoException &e) {
    std::cout << std::endl;
    std::cout << e.what() << std::endl;
  }

  //while (std::next_permutation(permutation, permutation + 4)) {
  //  try {
  //    auto this_result = run_contraction(x, y, permutation);
  //    if (equals(this_result, ground_truth)) {
  //      std::cout << "Correctness check passed!" << std::endl;
  //    } else {
  //      std::cout << "Correctness check failed!" << std::endl;
  //    }
  //  } catch (TacoException &e) {
  //    std::cout << std::endl;
  //    std::cout << e.what() << std::endl;
  //  }
  //  for (int i = 0; i < 4; i++) {
  //    std::cout << permutation[i] << " ";
  //  }
  //}
}
