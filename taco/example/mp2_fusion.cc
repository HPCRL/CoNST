#include "taco.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <sys/stat.h>
#include <taco/storage/file_io_tns.h>
#include <vector>
#include "3c_filter_fused.hpp"
#include "3c_nofilter_fused.hpp"
#include "util.hpp"

using namespace taco;

// const int MO = 100;
// const int PAO = 300;
// const int PAO_HAT = 300;
// const int AUX = 700;

//const int MO = 25;
//const int PAO = 75;
//const int PAO_HAT = 75;
//const int AUX = 175;


// void noncov_mp2_fused(Tensor<double> Int, Tensor<double> C, Tensor<double>
// Phat,
//                       Tensor<double> X, Tensor<double> L) {
//   IndexVar nu("nu");
//   IndexVar muhat("muhat");
//   IndexVar k("k");
//   IndexVar mu("mu");
//   IndexVar i("i");
//   TensorVar Int_var = Int.getTensorVar();
//   std::cout<<"Shape of int is "<<Int_var.getType().getShape()<<std::endl;
//   TensorVar C_var = C.getTensorVar();
//   TensorVar Phat_var = Phat.getTensorVar();
//   TensorVar X_var = X.getTensorVar();
//   TensorVar L_var = L.getTensorVar();
//   std::cout<<"Shape of L is "<<L_var.getType().getShape()<<std::endl;
//   TensorVar IntC =
//       TensorVar("IntC", Type(Int.getComponentType(), {PAO}), {taco::dense});
//   TensorVar IntCPhat =
//       TensorVar("IntCPhat", Type(Int.getComponentType(), {PAO_HAT}),
//       {taco::dense});
//   auto fused_ir = forall(
//       k,
//       forall(
//           i,
//           where(forall(muhat,
//                        X_var(k, i, muhat) += IntCPhat(muhat) * L_var(k, i)),
//                 where(forall(mu, forall(muhat, IntCPhat(muhat) +=
//                                                IntC(mu) * Phat_var(mu,
//                                                muhat))),
//                       forall(nu, forall(mu, IntC(mu) += Int_var(k, nu, mu) *
//                                                         C_var(i, nu)))))));
//   ;
//   X(k, i, muhat) = Int(k, nu, mu) * C(i, nu) * Phat(mu, muhat) * L(k, i);
//   X.compile(fused_ir);
//   auto start = std::chrono::high_resolution_clock::now();
//   X.assemble();
//   X.compute();
//   auto end = std::chrono::high_resolution_clock::now();
//   std::chrono::duration<double, std::milli> elapsed = end - start;
//   std::cout << "Time noncov_mp2_fused:  " << elapsed.count() << " ms "
//             << std::endl;
//   //write("fused_3cint.tns", X);
// }

void noncov_mp2_unfused(Tensor<double> Int, Tensor<double> C,
                        Tensor<double> Phat, Tensor<double> X,
                        Tensor<double> L) {
  IndexVar nu("nu");
  IndexVar muhat("muhat");
  IndexVar k("k");
  IndexVar mu("mu");
  IndexVar i("i");
  Tensor<double> IntC =
      Tensor<double>("IntC", {AUX, MO, PAO}, {Dense, Sparse, Sparse});
  Tensor<double> IntCPhat =
      Tensor<double>("IntCPhat", {PAO_HAT, AUX, MO}, {Dense, Sparse, Sparse});
  IntC(k, i, mu) = Int(k, mu, nu) * C(i, nu);
  IntCPhat(muhat, k, i) = IntC(k, i, mu) * Phat(muhat, mu);
  X(muhat, k, i) = IntCPhat(muhat, k, i) * L(k, i);
  IntC.compile();
  IntCPhat.compile();
  X.compile();
  auto start = std::chrono::high_resolution_clock::now();
  IntC.assemble();
  IntCPhat.assemble();
  X.assemble();
  IntC.compute();
  IntCPhat.compute();
  X.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time noncov_mp2_unfused:  " << elapsed.count() << " ms "
            << std::endl;
  // write("data_3cent/X_unfused.tns", X);
}

void noncov_mp2_nary(Tensor<double> Int, Tensor<double> C, Tensor<double> Phat,
                     Tensor<double> X, Tensor<double> L) {
  IndexVar nu("nu");
  IndexVar muhat("muhat");
  IndexVar k("k");
  IndexVar mu("mu");
  IndexVar i("i");
  Phat = Phat.transpose({1, 0});
  Phat.setName("Phat");

  X(k, i, muhat) = Int(k, nu, mu) * C(i, nu) * Phat(muhat, mu) * L(k, i);
  X.compile();
  auto start = std::chrono::high_resolution_clock::now();
  X.assemble();
  X.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time noncov_mp2_nary:  " << elapsed.count() << " ms "
            << std::endl;
  //write("nary_3cint.tns", X);
}

// void nofilter_fused(Tensor<double> Int, Tensor<double> C, Tensor<double> X,
//                     Tensor<double> Phat) {
//   IndexVar i("i");
//   IndexVar mu("mu");
//   IndexVar muhat("muhat");
//   IndexVar nu("nu");
//   IndexVar k("k");
//   TensorVar Int_var = Int.getTensorVar();
//   TensorVar C_var = C.getTensorVar();
//   TensorVar X_var = X.getTensorVar();
//   Phat = Phat.transpose({1, 0});
//   Phat.setName("Phat");
//   TensorVar Phat_var = Phat.getTensorVar();
//   TensorVar IntC =
//       TensorVar("IntC", Type(Int.getComponentType(), {PAO}), {taco::dense});
//   auto fused_ir = forall(
//       k,
//       forall(i, where(forall(muhat, forall(mu, X_var(k, i, muhat) +=
//                                                IntC(mu) * Phat_var(muhat,
//                                                mu))),
//                       forall(nu, forall(mu, IntC(mu) += Int_var(k, nu, mu) *
//                                                         C_var(i, nu))))));
//   X(k, i, muhat) = Int(k, nu, mu) * C(i, nu) * Phat(muhat, mu);
//   X.compile(fused_ir);
//   auto start = std::chrono::high_resolution_clock::now();
//   X.assemble();
//   X.compute();
//   auto end = std::chrono::high_resolution_clock::now();
//   std::chrono::duration<double, std::milli> elapsed = end - start;
//   std::cout << "Time nofilter_fused:  " << elapsed.count() << " ms "
//             << std::endl;
//   //write("data_3cent_real/threec_int.tns", X);
// }

// void nofilter_unfused(Tensor<double> Int, Tensor<double> C, Tensor<double> X,
//                       Tensor<double> Phat) {
//   IndexVar i("i");
//   IndexVar mu("mu");
//   IndexVar muhat("muhat");
//   IndexVar nu("nu");
//   IndexVar k("k");
//   Tensor<double> IntC =
//       Tensor<double>("IntC", {AUX, MO, PAO}, {Dense, Sparse, Sparse});
//   IntC(k, i, mu) = Int(k, nu, mu) * C(i, nu);
//   X(k, i, muhat) = IntC(k, i, mu) * Phat(mu, muhat);
//   IntC.compile();
//   X.compile();
//   auto start = std::chrono::high_resolution_clock::now();
//   IntC.assemble();
//   IntC.compute();
//   X.assemble();
//   X.compute();
//   auto end = std::chrono::high_resolution_clock::now();
//   std::chrono::duration<double, std::milli> elapsed = end - start;
//   std::cout << "Time nofilter_unfused:  " << elapsed.count() << " ms "
//             << std::endl;
//   // write("data_3cent/X_nofilter_unf.tns", X);
// }

void nofilter_nary(Tensor<double> Int, Tensor<double> C, Tensor<double> X,
                   Tensor<double> Phat) {
  IndexVar i("i");
  IndexVar mu("mu");
  IndexVar muhat("muhat");
  IndexVar nu("nu");
  IndexVar k("k");
  Phat = Phat.transpose({1, 0});
  Phat.setName("Phat");
  X(k, i, muhat) = Int(k, nu, mu) * C(i, nu) * Phat(muhat, mu);
  X.compile();

  auto start = std::chrono::high_resolution_clock::now();
  X.assemble();
  X.compute();
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> elapsed = end - start;
  std::cout << "Time nofilter_nary:  " << elapsed.count() << " ms "
            << std::endl;
  //write("threec_int_nary.tns", X);
}

int main() {
  source_env_vars_mp2(MO, PAO, PAO_HAT, AUX);
  // Tensor<double> Int("Int", {PAO, PAO, AUX}, {Dense, Sparse, Sparse});
  Tensor<double> Int = read("data_3cent_real/Int.tns", {Dense, Sparse, Sparse});
  Int.pack();
  // Int = Int.transpose({0, 2, 1});
  Int.setName("Int");

  // Tensor<double> L("L", {AUX, MO}, {Dense, Sparse});
  Tensor<double> L = read("data_3cent_real/L.tns", {Dense, Sparse});
  L.pack();
  L.setName("L");

  Tensor<double> Phat("Phat", {PAO, PAO_HAT}, {Dense, Sparse});
  Phat = read("data_3cent_real/Phat.tns", {Dense, Sparse});
  Phat.pack();
  Phat.setName("Phat");

  Tensor<double> C("C", {PAO, MO}, {Dense, Sparse});
  C = read("data_3cent_real/C.tns", {Dense, Sparse});
  Phat.pack();
  // C = C.transpose({1, 0});
  C.setName("C");

  Tensor<double> X("X", {AUX, MO, PAO_HAT}, {Dense, Sparse, Sparse});
  filter_fused(Int, C, Phat, X, L);
  // Tensor<double> X_unf("X", {PAO_HAT, AUX, MO}, {Dense, Sparse, Sparse});
  // noncov_mp2_unfused(Int, C, Phat, X_unf, L);
  //Int = Int.transpose({2, 0, 1});
  //C = C.transpose({1, 0});
  //Tensor<double> X_nary("X", {AUX, MO, PAO_HAT}, {Dense, Sparse, Sparse});
  //noncov_mp2_nary(Int, C, Phat, X_nary, L);
  //Tensor<double> X_nofilter("X", {MO, AUX, PAO_HAT}, {Dense, Sparse, Sparse});
  //nofilter_fused(Int, C, X_nofilter, Phat);
  //Tensor<double> X_nofilter_unf("X", {AUX, MO, PAO_HAT},
  //                              {Dense, Sparse, Sparse});
  //nofilter_unfused(Int, C, X_nofilter_unf, Phat);
  //Int = Int.transpose({2, 0, 1});
  //C = C.transpose({1, 0});
  //Tensor<double> X_nofilter_nary("X", {AUX, MO, PAO_HAT},
  //                               {Dense, Sparse, Sparse});
  //nofilter_nary(Int, C, X_nofilter_nary, Phat);
  return 0;
}
