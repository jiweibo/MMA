#include "common/common.h"

template <typename T1, typename T2=T1>
class CommonGemm {
  CommonGemm(T1* a, T1* b, T2* c, int lda, int ldb, int ldc) :
    a_(a),
    b_(b),
    c_(c),
    lda_(lda),
    ldb_(ldb),
    ldc_(ldc) {}

  

 protected:
  T1* a_;
  T1* b_;
  T2* c_;

  int lda_;
  int ldb_;
  int ldc_;
};