//
//  LBFGSB0.hpp
//  main
//
//

#ifndef LBFGSB0_hpp
#define LBFGSB0_hpp

#include <array>

#include <deal.II/lac/sparse_direct.h>

#include "Traits.h"

#include "BlockVectorWrapper.h"
#include "BlockSparseMatrixWrapper.h"

#include "BlockDesc.h"

#include "InverseMatrix.h"
#include "MPIPreconditionerGen.h"
#include "MPICGSolver.h"

namespace PhaseField_monolithic {

enum class SolverType
{
    Direct, CG
};

struct Tol
{
    const unsigned int nIters;
    const double tol;
    Tol(const unsigned int nIters,
        const double tol);
        
};

template <typename LATraits>
class LBFGSB0
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
    
    const std::array<Tol, 3> __tolList;
    
    const BlockDesc&        __blockDesc;
    
    
    void __directSolve(BVector & LBFGS_r_vector,
                       BVector & LBFGS_q_vector,
                       BSMatrix& tangentMatrix);
    void __cgSolve(BVector & LBFGS_r_vector,
                   BVector & LBFGS_q_vector,
                   BSMatrix& tangentMatrix);
    
public:
    
    virtual ~LBFGSB0() = default;
    
    
    LBFGSB0(const SolverType&   type,
                const double        cg_u_tol,
                const double        cg_d_tol,
                const double        cg_T_tol,
                const BlockDesc&    blockDesc);
    
    void solve(BVector & LBFGS_r_vector,
               BVector & LBFGS_q_vector,
               BSMatrix& tangentMatrix);
    
};





}
#endif /* LBFGSB0_hpp */
