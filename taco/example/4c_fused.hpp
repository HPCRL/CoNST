#include "taco.h"
using namespace taco;
#include "util.hpp"
void const_3c_to_4c(Tensor<double> I1_disk, Tensor<double> D_disk,
                    Tensor<double> Result_disk, Tensor<double> I2_disk) {
  IndexVar nuhat("nuhat");
  IndexVar muhat("muhat");
  IndexVar i("i");
  IndexVar j("j");
  IndexVar l("l");
  IndexVar l0("l0");
  IndexVar k("k");
  Tensor<double> I1 = getCSFOrder(I1_disk, {k, i, muhat}, {muhat, i, k});
  Tensor<double> D = D_disk;
  Tensor<double> Result = Result_disk;
  Tensor<double> I2 = getCSFOrder(I2_disk, {l, j, nuhat}, {nuhat, j, l});
  TensorVar I1_var = I1.getTensorVar();
  TensorVar D_var = D.getTensorVar();
  TensorVar Result_var = Result.getTensorVar();
  TensorVar I2_var = I2.getTensorVar();
  TensorVar I1D = TensorVar("I1D", Type(Float64, {AUX}), {taco::dense});
  auto fused_ir = forall(
      muhat,
      forall(i,
             where(forall(nuhat,
                          forall(j, forall(l0, Result_var(muhat, i, nuhat, j) +=
                                              I1D(l0) * I2_var(nuhat, j, l0)))),
                   forall(k, forall(l, I1D(l) +=
                                       I1_var(muhat, i, k) * D_var(k, l))))));
  ;
  auto par_fused_ir = fused_ir
                          .parallelize(l, ParallelUnit::CPUThread,
                                       OutputRaceStrategy::NoRaces);
  // par_fused_ir = par_fused_ir.parallelize(muhat, ParallelUnit::CPUThread,
  // OutputRaceStrategy::IgnoreRaces);
  Result(muhat, i, nuhat, j) += I1(muhat, i, k) * D(k, l) * I2(nuhat, j, l);
  std::cout << "Fused IR: " << par_fused_ir << std::endl;
  //Result.compile(fused_ir);
  Result.compile(par_fused_ir);
  auto start = std::chrono::high_resolution_clock::now();
  Result.assemble();
  Result.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time const_3c_to_4c:  " << elapsed.count() << " ms "
            << std::endl;
  //write("4centered_const_par.tns", Result);
}
