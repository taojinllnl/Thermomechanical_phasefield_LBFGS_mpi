//
//  LBFGSSolver.hpp
//  main
//
//

#ifndef LBFGSSolver_hpp
#define LBFGSSolver_hpp

#include "Traits.h"

#include "BlockVectorWrapper.h"
#include "BlockSparseMatrixWrapper.h"

namespace la {

template <typename LATraits>
class LBFGSSolver
{
public:
    using BSMatrix = BlockSparseMatrixWrapper<LATraits>;
    using BVector  = BlockVectorWrapper<LATraits>;
    
    virtual ~LBFGSSolver() = default;
    
    void solve();
    
};





}
#endif /* LBFGSSolver_hpp */
