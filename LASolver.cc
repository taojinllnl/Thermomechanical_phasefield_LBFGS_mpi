//
//  LASolver.cpp
//  main
//
//

#include "LASolver.h"

using namespace PhaseField_monolithic;
using namespace dealii;

Tol::Tol(const unsigned int nIters,
         const double tol)
: nIters(nIters)
, tol(tol)
{}





template <typename LATraits>
LASolver<LATraits>
::LASolver(const SolverType&   type,
           const double        cg_u_tol,
           const double        cg_d_tol,
           const double        cg_T_tol,
           const BlockDesc&    blockDesc)
: __type(type)
, __cg_u_tol(cg_u_tol)
, __cg_d_tol(cg_d_tol)
, __cg_T_tol(cg_T_tol)
, __u_group_ID(blockDesc.ithGroupID("displacement"))
, __d_group_ID(blockDesc.ithGroupID("phase-field"))
, __T_group_ID(blockDesc.ithGroupID("temperature"))
, __tolList({
    Tol(1e6, cg_u_tol),
    Tol(1e6, cg_d_tol),
    Tol(1e6, cg_T_tol)
})
, __blockDesc(blockDesc)
{}





template <typename LATraits>
void
LASolver<LATraits>::solve(BVector & LBFGS_r_vector,
                          const BVector & LBFGS_q_vector,
                          const BSMatrix& tangent_matrix)
{
    if (__type == SolverType::Direct) {
        __directSolve(LBFGS_r_vector, LBFGS_q_vector, tangent_matrix);
    } else {
        __cgSolve(LBFGS_r_vector, LBFGS_q_vector, tangent_matrix);
    }
}



template <typename LATraits>
void
LASolver<LATraits>::__directSolve(BVector & LBFGS_r_vector,
                                  const BVector & LBFGS_q_vector,
                                  const BSMatrix& tangent_matrix)
{
    using namespace dealii;
    if constexpr (std::is_same_v<typename LATraits::TMTag, ::la::TagSerial>) {
        /*
         SparseDirectUMFPACK A_direct;
         A_direct.initialize(m_tangent_matrix);
         A_direct.vmult(LBFGS_r_vector,
         LBFGS_q_vector);
         */
        
        // Performing LU decomposition on each block is much faster than
        // performing LU decomposition on the whole system
        
        //        {
        //            SparseDirectUMFPACK A_direct_u;
        //            A_direct_u.initialize(tangent_matrix.block(__u_group_ID, __u_group_ID));
        //            A_direct_u.vmult(LBFGS_r_vector.block(__u_group_ID),
        //                             LBFGS_q_vector.block(__u_group_ID));
        //        }
        //
        //        {
        //            SparseDirectUMFPACK A_direct_d;
        //            A_direct_d.initialize(tangent_matrix.block(__d_group_ID, __d_group_ID));
        //            A_direct_d.vmult(LBFGS_r_vector.block(__d_group_ID),
        //                             LBFGS_q_vector.block(__d_group_ID));
        //        }
        //
        //        {
        //            SparseDirectUMFPACK A_direct_t;
        //            A_direct_t.initialize(tangent_matrix.block(__T_group_ID, __T_group_ID));
        //            A_direct_t.vmult(LBFGS_r_vector.block(__T_group_ID),
        //                             LBFGS_q_vector.block(__T_group_ID));
        //        }
        for (const unsigned int ithGroup : __blockDesc.groupIDs())
        {
            SparseDirectUMFPACK A_direct;
            A_direct.initialize(tangent_matrix.block(ithGroup, ithGroup));
            A_direct.vmult(LBFGS_r_vector.block(ithGroup),
                           LBFGS_q_vector.block(ithGroup));
        }
    } else if constexpr (std::is_same_v<typename LATraits::TMTag, ::la::TagPETSc>) {
        
        using PrecJacobi = dealii::PETScWrappers::PreconditionBlockJacobi;
        using PrecILU    = dealii::PETScWrappers::PreconditionILU;
        using PrecICC    = dealii::PETScWrappers::PreconditionICC;
        using PrecPSails = dealii::PETScWrappers::PreconditionParaSails;
        using PrecSOR    = dealii::PETScWrappers::PreconditionSOR;
        using PrecSSOR   = dealii::PETScWrappers::PreconditionSSOR;
        using PrecShell  = dealii::PETScWrappers::PreconditionShell;
        using PrecNone   = dealii::PETScWrappers::PreconditionNone;
        using MatBlock   = typename LATraits::MatrixBlock;
        
        using InverseMatrix = InverseMatrix<MatBlock, PrecJacobi>;
        
        for (const unsigned int ithGroup : __blockDesc.groupIDs())
        {
            SolverControl solver_control(__tolList[ithGroup].nIters,
                                         __tolList[ithGroup].tol);
            PrecJacobi prec;
            prec.initialize(tangent_matrix.block(ithGroup, ithGroup));
            
            
            InverseMatrix A_direct(tangent_matrix.block(__u_group_ID, __u_group_ID),
                                   prec);
            A_direct.vmult(solver_control,
                           LBFGS_r_vector.block(__u_group_ID),
                           LBFGS_q_vector.block(__u_group_ID));
            
        }
        
        //        SolverControl solver_control_uu(1e6, __cg_u_tol);
        //        SolverControl solver_control_dd(1e6, __cg_d_tol);
        //        SolverControl solver_control_TT(1e6, __cg_T_tol);
        //        MPIPreconditionerGen<LATraits> precond_uu(m_parameters.m_preconditioner_type,
        //                                           tangent_matrix.block(__u_group_ID, __u_group_ID));
        //
        //        MPIPreconditionerGen<LATraits> precond_dd(m_parameters.m_preconditioner_type,
        //                                                       tangent_matrix.block(__d_group_ID, __d_group_ID));
        //        {
        //            InverseMatrix<LA::MPI::SparseMatrix, LA::PreconditionBase> A_direct_uu(tangent_matrix.block(__u_group_ID, __u_group_ID), precond_uu.preconditioner());
        //            A_direct_uu.vmult(solver_control_uu,
        //                              LBFGS_r_vector.block(__u_group_ID),
        //                              LBFGS_q_vector.block(__u_group_ID));
        //        }
    } else if constexpr (std::is_same_v<typename LATraits::TMTag, ::la::TagTrilinos>) {
        using PrecJacobi = dealii::TrilinosWrappers::PreconditionBlockJacobi;
        using PrecILU    = dealii::TrilinosWrappers::PreconditionILU;
        using PrecIC     = dealii::TrilinosWrappers::PreconditionIC;
        using PrecILUT   = dealii::TrilinosWrappers::PreconditionILUT;
        using PrecSOR    = dealii::TrilinosWrappers::PreconditionSOR;
        using PrecSSOR   = dealii::TrilinosWrappers::PreconditionSSOR;
        using PrecShebs  = dealii::TrilinosWrappers::PreconditionChebyshev;
        using PrecI      = dealii::TrilinosWrappers::PreconditionIdentity;
        using MatBlock   = typename LATraits::MatrixBlock;
        
        using InverseMatrix = InverseMatrix<MatBlock, PrecJacobi>;
        
        for (const unsigned int ithGroup : __blockDesc.groupIDs())
        {
            SolverControl solver_control(__tolList[ithGroup].nIters,
                                         __tolList[ithGroup].tol);
            PrecJacobi prec;
            prec.initialize(tangent_matrix.block(ithGroup, ithGroup));
            
            
            InverseMatrix A_direct(tangent_matrix.block(__u_group_ID, __u_group_ID),
                                   prec);
            A_direct.vmult(solver_control,
                           LBFGS_r_vector.block(__u_group_ID),
                           LBFGS_q_vector.block(__u_group_ID));
        }
        //
        //        {
        //
        //            ::LinearSolvers::InverseMatrix<LA::MPI::SparseMatrix, LA::PreconditionBase> A_direct_dd(m_tangent_matrix.block(m_d_dof, m_d_dof), precond_dd.preconditioner());
        //            A_direct_dd.vmult(solver_control_dd,
        //                              LBFGS_r_vector.block(m_d_dof),
        //                              LBFGS_q_vector.block(m_d_dof));
        //        }
    }
}

template <typename LATraits>
void
LASolver<LATraits>::__cgSolve(BVector & LBFGS_r_vector,
                              const BVector & LBFGS_q_vector,
                              const BSMatrix& tangent_matrix)
{
    using namespace dealii;
    if constexpr (std::is_same_v<typename LATraits::TMTag, ::la::TagSerial>) {
        /*
         SolverControl            solver_control(1e6, 1e-9);
         SolverCG<BlockVector<double>> cg(solver_control);
         
         PreconditionJacobi<BlockSparseMatrix<double>> preconditioner;
         preconditioner.initialize(m_tangent_matrix, 1.0);
         
         cg.solve(m_tangent_matrix,
         LBFGS_r_vector,
         LBFGS_q_vector,
         preconditioner);
         */
        //        SolverControl            solver_control_uu(1e6, __cg_u_tol);
        //        SolverCG<Vector<double>> cg_uu(solver_control_uu);
        //
        //        PreconditionJacobi<SparseMatrix<double>> preconditioner_uu;
        //        preconditioner_uu.initialize(tangent_matrix.block(__u_group_ID, __u_group_ID), 1.0);
        //        cg_uu.solve(tangent_matrix.block(__u_group_ID, __u_group_ID),
        //                    LBFGS_r_vector.block(__u_group_ID),
        //                    LBFGS_q_vector.block(__u_group_ID),
        //                    preconditioner_uu);
        //
        //        SolverControl            solver_control_dd(1e6, __cg_d_tol);
        //        SolverCG<Vector<double>> cg_dd(solver_control_dd);
        //
        //        PreconditionJacobi<SparseMatrix<double>> preconditioner_dd;
        //        preconditioner_dd.initialize(tangent_matrix.block(__d_group_ID, __d_group_ID), 1.0);
        //        cg_dd.solve(tangent_matrix.block(__d_group_ID, __d_group_ID),
        //                    LBFGS_r_vector.block(__d_group_ID),
        //                    LBFGS_q_vector.block(__d_group_ID),
        //                    preconditioner_dd);
        //
        //        SolverControl            solver_control_tt(1e6, __cg_T_tol);
        //        SolverCG<Vector<double>> cg_tt(solver_control_tt);
        //
        //        PreconditionJacobi<SparseMatrix<double>> preconditioner_tt;
        //        preconditioner_tt.initialize(tangent_matrix.block(__T_group_ID, __T_group_ID), 1.0);
        //        cg_tt.solve(tangent_matrix.block(__T_group_ID, __T_group_ID),
        //                    LBFGS_r_vector.block(__T_group_ID),
        //                    LBFGS_q_vector.block(__T_group_ID),
        //                    preconditioner_tt);
        
        for (const unsigned int ithGroup : __blockDesc.groupIDs()) {
            SolverControl            solver_control(1e6, ithGroup);
            SolverCG<Vector<double>> cg(solver_control);
            
            PreconditionJacobi<SparseMatrix<double>> preconditioner;
            preconditioner.initialize(tangent_matrix.block(ithGroup, ithGroup),
                                      1.0);
            
            cg.solve(tangent_matrix.block(ithGroup, ithGroup),
                     LBFGS_r_vector.block(ithGroup),
                     LBFGS_q_vector.block(ithGroup),
                     preconditioner);
        }
        
    } else if constexpr (std::is_same_v<typename LATraits::TMTag, ::la::TagPETSc>) {
        using PrecJacobi = dealii::PETScWrappers::PreconditionBlockJacobi;
        using PrecILU    = dealii::PETScWrappers::PreconditionILU;
        using PrecICC    = dealii::PETScWrappers::PreconditionICC;
        using PrecPSails = dealii::PETScWrappers::PreconditionParaSails;
        using PrecSOR    = dealii::PETScWrappers::PreconditionSOR;
        using PrecSSOR   = dealii::PETScWrappers::PreconditionSSOR;
        using PrecShell  = dealii::PETScWrappers::PreconditionShell;
        using PrecNone   = dealii::PETScWrappers::PreconditionNone;
        
        using MatBlock   = typename LATraits::MatrixBlock;
        
        using CGSolver   =  MPICGSolver<MatBlock, PETScWrappers::SolverCG>;
        
        for (const unsigned int ithGroup : __blockDesc.groupIDs())
        {
            PrecJacobi prec;
            prec.initialize(tangent_matrix.block(ithGroup, ithGroup));
            
            CGSolver cg(__tolList[ithGroup].tol,
                        __tolList[ithGroup].nIters);
            
            cg.solve(tangent_matrix.block(ithGroup, ithGroup),
                     LBFGS_r_vector.block(ithGroup),
                     LBFGS_q_vector.block(ithGroup),
                     prec);
        }
        
    } else if constexpr (std::is_same_v<typename LATraits::TMTag, ::la::TagTrilinos>) {
        using PrecJacobi = dealii::TrilinosWrappers::PreconditionBlockJacobi;
        using PrecILU    = dealii::TrilinosWrappers::PreconditionILU;
        using PrecIC     = dealii::TrilinosWrappers::PreconditionIC;
        using PrecILUT   = dealii::TrilinosWrappers::PreconditionILUT;
        using PrecSOR    = dealii::TrilinosWrappers::PreconditionSOR;
        using PrecSSOR   = dealii::TrilinosWrappers::PreconditionSSOR;
        using PrecShebs  = dealii::TrilinosWrappers::PreconditionChebyshev;
        using PrecI      = dealii::TrilinosWrappers::PreconditionIdentity;
        
        using MatBlock   = typename LATraits::MatrixBlock;
        
        using CGSolver   =  MPICGSolver<MatBlock, TrilinosWrappers::SolverCG>;
        
        for (const unsigned int ithGroup : __blockDesc.groupIDs())
        {
            PrecJacobi prec;
            prec.initialize(tangent_matrix.block(ithGroup, ithGroup));
            
            CGSolver cg(__tolList[ithGroup].tol,
                        __tolList[ithGroup].nIters);
            
            cg.solve(tangent_matrix.block(ithGroup, ithGroup),
                     LBFGS_r_vector.block(ithGroup),
                     LBFGS_q_vector.block(ithGroup),
                     prec);
        }
        
    }
}




template class PhaseField_monolithic::LASolver<la::Traits<la::TagSerial>>;
template class PhaseField_monolithic::LASolver<la::Traits<la::TagPETSc>>;
template class PhaseField_monolithic::LASolver<la::Traits<la::TagTrilinos>>;
