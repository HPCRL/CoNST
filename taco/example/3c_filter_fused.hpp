#include "taco.h"
using namespace taco;
#include "util.hpp"
void filter_const(Tensor<double> Int_disk, Tensor<double> C_disk,
                  Tensor<double> Phat_disk, Tensor<double> X_disk,
                  Tensor<double> L_disk) {
  IndexVar i("i");
  IndexVar muhat("muhat");
  IndexVar muhat0("muhat0");
  IndexVar nu("nu");
  IndexVar mu0("mu0");
  IndexVar mu("mu");
  IndexVar k("k");
  Tensor<double> Int = getCSFOrder(Int_disk, {mu, nu, k}, {k, nu, mu});
  Tensor<double> C = getCSFOrder(C_disk, {nu, i}, {i, nu});
  Tensor<double> Phat = Phat_disk;
  Tensor<double> X = X_disk;
  Tensor<double> L = L_disk;
  TensorVar Int_var = Int.getTensorVar();
  TensorVar C_var = C.getTensorVar();
  TensorVar Phat_var = Phat.getTensorVar();
  TensorVar X_var = X.getTensorVar();
  TensorVar L_var = L.getTensorVar();
  TensorVar IntC = TensorVar("IntC", Type(Float64, {PAO}), {taco::dense});
  TensorVar IntCPhat =
      TensorVar("IntCPhat", Type(Float64, {PAO}), {taco::dense});
  auto fused_ir = forall(
      k,
      forall(i,
             where(forall(muhat0,
                          X_var(k, i, muhat0) += IntCPhat(muhat0) * L_var(k, i)),
                   where(forall(mu0, forall(muhat,
                                            IntCPhat(muhat) +=
                                            IntC(mu0) * Phat_var(mu0, muhat))),
                         forall(nu, forall(mu, IntC(mu) += Int_var(k, nu, mu) *
                                                           C_var(i, nu)))))));
  ;
  X(k, i, muhat) += Int(k, nu, mu) * C(i, nu) * Phat(mu, muhat) * L(k, i);
  auto par_fused_ir = fused_ir
                          //.parallelize(muhat, ParallelUnit::CPUThread,
                          //             OutputRaceStrategy::NoRaces);
                          .parallelize(mu, ParallelUnit::CPUThread,
                                       OutputRaceStrategy::NoRaces);
  X.compile(fused_ir);
  auto start = std::chrono::high_resolution_clock::now();
  X.assemble();
  X.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time filter_const:  " << elapsed.count() << " ms " << std::endl;
}
