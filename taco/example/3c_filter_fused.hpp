#include "taco.h"
using namespace taco;
#include "util.hpp"
void filter_fused(Tensor<double> Int_disk, Tensor<double> C_disk,
                  Tensor<double> Phat_disk, Tensor<double> X_disk,
                  Tensor<double> L_disk) {
  IndexVar i("i");
  IndexVar muhat("muhat");
  IndexVar nu("nu");
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
      forall(
          i,
          where(forall(muhat,
                       X_var(k, i, muhat) += IntCPhat(muhat) * L_var(k, i)),
                where(forall(mu, forall(muhat, IntCPhat(muhat) +=
                                               IntC(mu) * Phat_var(mu, muhat))),
                      forall(nu, forall(mu, IntC(mu) += Int_var(k, nu, mu) *
                                                        C_var(i, nu)))))));
  ;
  X(k, i, muhat) += Int(k, nu, mu) * C(i, nu) * Phat(mu, muhat) * L(k, i);
  X.compile(fused_ir);
  auto start = std::chrono::high_resolution_clock::now();
  X.assemble();
  X.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time filter_fused:  " << elapsed.count() << " ms " << std::endl;
}
