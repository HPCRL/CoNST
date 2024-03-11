#include "taco.h"
using namespace taco;
#include "util.hpp"
void nofilter_const(Tensor<double> Int_disk, Tensor<double> C_disk,
                    Tensor<double> X_nofilter_disk, Tensor<double> Phat_disk) {
  IndexVar i("i");
  IndexVar muhat("muhat");
  IndexVar nu("nu");
  IndexVar mu("mu");
  IndexVar k("k");
  Tensor<double> Int = getCSFOrder(Int_disk, {mu, nu, k}, {k, nu, mu});
  Tensor<double> C = getCSFOrder(C_disk, {nu, i}, {i, nu});
  Tensor<double> X_nofilter = X_nofilter_disk;
  Tensor<double> Phat = getCSFOrder(Phat_disk, {mu, muhat}, {muhat, mu});
  TensorVar Int_var = Int.getTensorVar();
  TensorVar C_var = C.getTensorVar();
  TensorVar X_nofilter_var = X_nofilter.getTensorVar();
  TensorVar Phat_var = Phat.getTensorVar();
  TensorVar IntC = TensorVar("IntC", Type(Float64, {PAO}), {taco::dense});
  auto fused_ir = forall(
      i,
      forall(k, where(forall(muhat, forall(mu, X_nofilter_var(i, k, muhat) +=
                                               IntC(mu) * Phat_var(muhat, mu))),
                      forall(nu, forall(mu, IntC(mu) += Int_var(k, nu, mu) *
                                                        C_var(i, nu))))));
  ;
  X_nofilter(i, k, muhat) += Int(k, nu, mu) * C(i, nu) * Phat(muhat, mu);
  X_nofilter.compile(fused_ir);
  auto start = std::chrono::high_resolution_clock::now();
  X_nofilter.assemble();
  X_nofilter.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time nofilter_const:  " << elapsed.count() << " ms "
            << std::endl;
}
