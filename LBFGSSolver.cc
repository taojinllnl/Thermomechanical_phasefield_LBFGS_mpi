//
//  LBFGSSolver.cpp
//  main
//
//

#include "LBFGSSolver.h"

using namespace la;

template <typename LATraits>
void 
LBFGSSolver<LATraits>::solve()
{
//        if (m_parameters.m_type_linear_solver == "Direct")
//          {
//        /*
//        SparseDirectUMFPACK A_direct;
//        A_direct.initialize(m_tangent_matrix);
//        A_direct.vmult(LBFGS_r_vector,
//                   LBFGS_q_vector);
//                   */
//
//        // Performing LU decomposition on each block is much faster than
//        // performing LU decomposition on the whole system
//        SparseDirectUMFPACK A_direct_u;
//        A_direct_u.initialize(m_tangent_matrix.block(m_u_dof, m_u_dof));
//        A_direct_u.vmult(LBFGS_r_vector.block(m_u_dof),
//                     LBFGS_q_vector.block(m_u_dof));
//
//        SparseDirectUMFPACK A_direct_d;
//        A_direct_d.initialize(m_tangent_matrix.block(m_d_dof, m_d_dof));
//        A_direct_d.vmult(LBFGS_r_vector.block(m_d_dof),
//                     LBFGS_q_vector.block(m_d_dof));
//
//        SparseDirectUMFPACK A_direct_t;
//        A_direct_t.initialize(m_tangent_matrix.block(m_t_dof, m_t_dof));
//        A_direct_t.vmult(LBFGS_r_vector.block(m_t_dof),
//                     LBFGS_q_vector.block(m_t_dof));
//
//          }
//        else if (m_parameters.m_type_linear_solver == "CG")
//          {
//    /*
//        SolverControl            solver_control(1e6, 1e-9);
//        SolverCG<BlockVector<double>> cg(solver_control);
//
//        PreconditionJacobi<BlockSparseMatrix<double>> preconditioner;
//        preconditioner.initialize(m_tangent_matrix, 1.0);
//
//        cg.solve(m_tangent_matrix,
//             LBFGS_r_vector,
//             LBFGS_q_vector,
//             preconditioner);
//    */
//        SolverControl            solver_control_uu(1e6, m_parameters.m_cg_u_tol);
//        SolverCG<Vector<double>> cg_uu(solver_control_uu);
//
//        PreconditionJacobi<SparseMatrix<double>> preconditioner_uu;
//        preconditioner_uu.initialize(m_tangent_matrix.block(m_u_dof, m_u_dof), 1.0);
//        cg_uu.solve(m_tangent_matrix.block(m_u_dof, m_u_dof),
//                    LBFGS_r_vector.block(m_u_dof),
//                    LBFGS_q_vector.block(m_u_dof),
//                    preconditioner_uu);
//
//        SolverControl            solver_control_dd(1e6, m_parameters.m_cg_d_tol);
//        SolverCG<Vector<double>> cg_dd(solver_control_dd);
//
//        PreconditionJacobi<SparseMatrix<double>> preconditioner_dd;
//        preconditioner_dd.initialize(m_tangent_matrix.block(m_d_dof, m_d_dof), 1.0);
//        cg_dd.solve(m_tangent_matrix.block(m_d_dof, m_d_dof),
//                    LBFGS_r_vector.block(m_d_dof),
//                    LBFGS_q_vector.block(m_d_dof),
//                    preconditioner_dd);
//
//        SolverControl            solver_control_tt(1e6, m_parameters.m_cg_t_tol);
//        SolverCG<Vector<double>> cg_tt(solver_control_tt);
//
//        PreconditionJacobi<SparseMatrix<double>> preconditioner_tt;
//        preconditioner_tt.initialize(m_tangent_matrix.block(m_t_dof, m_t_dof), 1.0);
//        cg_tt.solve(m_tangent_matrix.block(m_t_dof, m_t_dof),
//                    LBFGS_r_vector.block(m_t_dof),
//                    LBFGS_q_vector.block(m_t_dof),
//                    preconditioner_tt);
//          }
//        else
//          {
//        AssertThrow(false,
//                    ExcMessage("Selected linear solver not implemented!"));
//          }
//

}




template class la::LBFGSSolver<la::Traits<la::TagSerial>>;
template class la::LBFGSSolver<la::Traits<la::TagPETSc>>;
template class la::LBFGSSolver<la::Traits<la::TagTrilinos>>;
