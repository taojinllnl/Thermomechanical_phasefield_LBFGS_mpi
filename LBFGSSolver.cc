//
//  LBFGSSolver.cpp
//  main
//
//

#include "LBFGSSolver.h"

using namespace PhaseField_monolithic;


template <typename LATraits>
LBFGSSolver<LATraits>
::LBFGSSolver(const SolverType&   type,
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
{}




template <typename LATraits>
void 
LBFGSSolver<LATraits>::solve(BVector & LBFGS_r_vector,
                             BVector & LBFGS_q_vector,
                             BSMatrix& tangent_matrix)
{
    if (__type == SolverType::Direct) {
        __directSolve(LBFGS_r_vector, LBFGS_q_vector, tangent_matrix);
    } else {
        __cgSolve(LBFGS_r_vector, LBFGS_q_vector, tangent_matrix);
    }
}



template <typename LATraits>
void
LBFGSSolver<LATraits>::__directSolve(BVector & LBFGS_r_vector,
                                     BVector & LBFGS_q_vector,
                                     BSMatrix& tangent_matrix)
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
        SparseDirectUMFPACK A_direct_u;
        A_direct_u.initialize(tangent_matrix.block(__u_group_ID, __u_group_ID));
        A_direct_u.vmult(LBFGS_r_vector.block(__u_group_ID),
                         LBFGS_q_vector.block(__u_group_ID));
        
        SparseDirectUMFPACK A_direct_d;
        A_direct_d.initialize(tangent_matrix.block(__d_group_ID, __d_group_ID));
        A_direct_d.vmult(LBFGS_r_vector.block(__d_group_ID),
                         LBFGS_q_vector.block(__d_group_ID));
        
        SparseDirectUMFPACK A_direct_t;
        A_direct_t.initialize(tangent_matrix.block(__T_group_ID, __T_group_ID));
        A_direct_t.vmult(LBFGS_r_vector.block(__T_group_ID),
                         LBFGS_q_vector.block(__T_group_ID));
    } else {
        
    }
}

template <typename LATraits>
void
LBFGSSolver<LATraits>::__cgSolve(BVector & LBFGS_r_vector,
                                 BVector & LBFGS_q_vector,
                                 BSMatrix& tangent_matrix)
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
        SolverControl            solver_control_uu(1e6, __cg_u_tol);
        SolverCG<Vector<double>> cg_uu(solver_control_uu);
        
        PreconditionJacobi<SparseMatrix<double>> preconditioner_uu;
        preconditioner_uu.initialize(tangent_matrix.block(__u_group_ID, __u_group_ID), 1.0);
        cg_uu.solve(tangent_matrix.block(__u_group_ID, __u_group_ID),
                    LBFGS_r_vector.block(__u_group_ID),
                    LBFGS_q_vector.block(__u_group_ID),
                    preconditioner_uu);
        
        SolverControl            solver_control_dd(1e6, __cg_d_tol);
        SolverCG<Vector<double>> cg_dd(solver_control_dd);
        
        PreconditionJacobi<SparseMatrix<double>> preconditioner_dd;
        preconditioner_dd.initialize(tangent_matrix.block(__d_group_ID, __d_group_ID), 1.0);
        cg_dd.solve(tangent_matrix.block(__d_group_ID, __d_group_ID),
                    LBFGS_r_vector.block(__d_group_ID),
                    LBFGS_q_vector.block(__d_group_ID),
                    preconditioner_dd);
        
        SolverControl            solver_control_tt(1e6, __cg_T_tol);
        SolverCG<Vector<double>> cg_tt(solver_control_tt);
        
        PreconditionJacobi<SparseMatrix<double>> preconditioner_tt;
        preconditioner_tt.initialize(tangent_matrix.block(__T_group_ID, __T_group_ID), 1.0);
        cg_tt.solve(tangent_matrix.block(__T_group_ID, __T_group_ID),
                    LBFGS_r_vector.block(__T_group_ID),
                    LBFGS_q_vector.block(__T_group_ID),
                    preconditioner_tt);
    } else {
        
    }
}




template class PhaseField_monolithic::LBFGSSolver<la::Traits<la::TagSerial>>;
template class PhaseField_monolithic::LBFGSSolver<la::Traits<la::TagPETSc>>;
template class PhaseField_monolithic::LBFGSSolver<la::Traits<la::TagTrilinos>>;
