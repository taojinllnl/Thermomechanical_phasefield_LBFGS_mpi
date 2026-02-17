//
//  LASolver.hpp
//  main
//
//

#ifndef LASolver_hpp
#define LASolver_hpp

#include <variant>
#include <array>

#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>

#include "Traits.h"

#include "BlockVectorWrapper.h"
#include "BlockSparseMatrixWrapper.h"

#include "BlockDesc.h"
#include "MPIInfo.h"

#include "MPICGSolver.h"

namespace PhaseField_monolithic {


/**
 *
 * The class `LASolver` is an integrated interface for selection of different types of linear algebra solvers in serial or MPI modes.
 * It supports:
 * - Sparse direct solver
 * - Iterative solver (conjugate gradient solver)
 *
 *
 */

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
class LASolver
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
    const MPIInfo&          __mpiInfo;
    
    void __directSolve(BVector & LBFGS_r_vector,
                       const BVector & LBFGS_q_vector,
                       const BSMatrix& tangentMatrix);
    void __cgSolve(BVector & LBFGS_r_vector,
                   const BVector & LBFGS_q_vector,
                   const BSMatrix& tangentMatrix);
    
public:
    
    virtual ~LASolver() = default;
    
    
    LASolver(const SolverType&   type,
             const double        cg_u_tol,
             const double        cg_d_tol,
             const double        cg_T_tol,
             const BlockDesc&    blockDesc,
             const MPIInfo&      mpiInfo);
    
    void solve(BVector & LBFGS_r_vector,
               const BVector & LBFGS_q_vector,
               const BSMatrix& tangentMatrix);
    
};





}
#endif /* LASolver_hpp */
