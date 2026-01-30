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

#include "InverseMatrix.h"
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
struct PrecSelector;

template <>
struct PrecSelector<::la::Traits<::la::TagPETSc>>
{
    
    using PrecJacobi = dealii::PETScWrappers::PreconditionBlockJacobi;
    using PrecILU    = dealii::PETScWrappers::PreconditionILU;
    using PrecICC    = dealii::PETScWrappers::PreconditionICC;
    using PrecPSails = dealii::PETScWrappers::PreconditionParaSails;
    using PrecSOR    = dealii::PETScWrappers::PreconditionSOR;
    using PrecSSOR   = dealii::PETScWrappers::PreconditionSSOR;
//        using PrecShell  = dealii::PETScWrappers::PreconditionShell;
    using PrecNone   = dealii::PETScWrappers::PreconditionNone;

    
    enum class Type {
      none, jacobi, ilu, icc, parasails, sor, ssor
    };
    
    const Type type;
    
    static Type parse(const std::string& str);
    
    PrecSelector(const std::string& str);
    
//    dealii::PETScWrappers::PreconditionBase&
};


template <>
struct PrecSelector<::la::Traits<::la::TagTrilinos>>
{
    
    using PrecJacobi = dealii::TrilinosWrappers::PreconditionBlockJacobi;
    using PrecILU    = dealii::TrilinosWrappers::PreconditionILU;
    using PrecIC     = dealii::TrilinosWrappers::PreconditionIC;
    using PrecILUT   = dealii::TrilinosWrappers::PreconditionILUT;
    using PrecSOR    = dealii::TrilinosWrappers::PreconditionSOR;
    using PrecSSOR   = dealii::TrilinosWrappers::PreconditionSSOR;
    using PrecShebs  = dealii::TrilinosWrappers::PreconditionChebyshev;
    using PrecI      = dealii::TrilinosWrappers::PreconditionIdentity;
    
    using PrecCollect = std::variant<PrecJacobi, PrecILU, PrecIC, PrecILUT, PrecSOR, PrecSSOR, PrecShebs, PrecI>;
    
    enum class Type {
      indentity, jacobi, ilu, ic, ilut, shebs, sor, ssor
    };
    
    const Type type;
    
    
    static Type parse(const std::string& str);
    PrecSelector(const std::string& str);
    
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
