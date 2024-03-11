#ifndef UTIL_HPP
#define UTIL_HPP
#include "taco.h"
#include <iostream>
#include <vector>
using namespace taco;
int MO = -1;
int PAO = -1;
int PAO_HAT = -1;
int AUX = -1;
int Rank = -1;

std::string parseExperiment(std::string exp_size) {
  if (exp_size == "small") {
    MO = 25;
    PAO = 75;
    PAO_HAT = 75;
    AUX = 175;
    return "data_qc_small";
  } else if (exp_size == "medium") {
    MO = 100;
    PAO = 300;
    PAO_HAT = 300;
    AUX = 700;
    return "data_qc_med";
  } else if (exp_size == "large") {
    MO = 100;
    PAO = 500;
    PAO_HAT = 500;
    AUX = 900;
    return "data_qc_large";
  } else {
    std::cout << "Invalid experiment size" << std::endl;
    return "";
  }
}

Tensor<double> getCSFOrder(Tensor<double> inp,
                           std::vector<IndexVar> currentOrder,
                           std::vector<IndexVar> newOrder) {
  // call a transpose such that the currentOrder is transformed to newOrder
  std::vector<int> transposeVec;
  for (auto &i : newOrder) {
    auto position = std::find(currentOrder.begin(), currentOrder.end(), i);
    transposeVec.push_back(position - currentOrder.begin());
  }
  return inp.transpose(transposeVec);
}
#endif
