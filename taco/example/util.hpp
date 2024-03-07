#ifndef UTIL_HPP
#define UTIL_HPP
#include "taco.h"
#include <vector>
#include <iostream>
using namespace taco;
Tensor<double> getCSFOrder(Tensor<double> inp, std::vector<IndexVar> currentOrder, std::vector<IndexVar> newOrder){
    // call a transpose such that the currentOrder is transformed to newOrder
    std::vector<int> transposeVec;
    for(auto &i: newOrder){
        auto position = std::find(currentOrder.begin(), currentOrder.end(), i);
        transposeVec.push_back(position - currentOrder.begin());
    }
    return inp.transpose(transposeVec);
}
#endif
