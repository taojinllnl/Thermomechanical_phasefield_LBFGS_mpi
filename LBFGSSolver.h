//
//  LBFGSSolver.hpp
//  main
//
//

#ifndef LBFGSSolver_hpp
#define LBFGSSolver_hpp

#include <deal.II/lac/sparse_direct.h>

#include "Traits.h"

#include "BlockVectorWrapper.h"
#include "BlockSparseMatrixWrapper.h"

#include "BlockDesc.h"


namespace PhaseField_monolithic {

enum class SolverType
{
    Direct, CG
};

template <typename LATraits>
class LBFGSSolver
{
public:
    using BSMatrix  = ::la::BlockSparseMatrixWrapper<LATraits>;
    using BVector   = ::la::BlockVectorWrapper<LATraits>;
    
private:
    const SolverType        __type;
    const double            __cg_u_tol;
    const double            __cg_d_tol;
    const double            __cg_T_tol;
    
    const unsigned int      __u_group_ID;
    const unsigned int      __d_group_ID;
    const unsigned int      __T_group_ID;
    
    
    void __directSolve(BVector & LBFGS_r_vector,
                       BVector & LBFGS_q_vector,
                       BSMatrix& tangentMatrix);
    void __cgSolve(BVector & LBFGS_r_vector,
                   BVector & LBFGS_q_vector,
                   BSMatrix& tangentMatrix);
    
public:
    
    virtual ~LBFGSSolver() = default;
    
    
    LBFGSSolver(const SolverType&   type,
                const double        cg_u_tol,
                const double        cg_d_tol,
                const double        cg_T_tol,
                const BlockDesc&    blockDesc);
    
    void solve(BVector & LBFGS_r_vector,
               BVector & LBFGS_q_vector,
               BSMatrix& tangentMatrix);
    
};





}
#endif /* LBFGSSolver_hpp */
