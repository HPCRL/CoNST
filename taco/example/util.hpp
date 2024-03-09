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

void source_env_vars_mp2(int &MO, int &PAO, int &PAO_HAT, int &AUX) {
  std::string mo = std::getenv("MO");
  std::string pao = std::getenv("PAO");
  std::string pao_hat = std::getenv("PAO_HAT");
  std::string aux = std::getenv("AUX");
  if (mo != "") {
    MO = std::stoi(mo);
  }
  if (pao != "") {
    PAO = std::stoi(pao);
  }
  if (pao_hat != "") {
    PAO_HAT = std::stoi(pao_hat);
  }
  if (aux != "") {
    AUX = std::stoi(aux);
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
