//
//  Traits.h
//  main
//
//

#ifndef Traits_h
#define Traits_h


#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/solution_transfer.h>

namespace la {

struct TagSerial    {};
struct TagPETSc     {};
struct TagTrilinos  {};



// templated class for different scenarios for class types
template <typename BackendTag>
struct Traits;

// serial mode
template <>
struct Traits<TagSerial>
{
    using TMTag    = TagSerial;
    using Vector   = ::dealii::LinearAlgebraDealII::BlockVector;
    using Matrix   = ::dealii::LinearAlgebraDealII::BlockSparseMatrix;
    using IndexSet = ::dealii::IndexSet;
    
    using MatrixBlock   = ::dealii::LinearAlgebraDealII::SparseMatrix;
    using VectorBlock   = ::dealii::LinearAlgebraDealII::Vector;
    
    static constexpr bool IS_MPI = false;
};
}

// alias for serial Triangulation
template <int dim, int spacedim = dim>
using RTria = ::dealii::Triangulation<dim, spacedim>;




// MPI mode with PETSc
#ifdef DEAL_II_WITH_PETSC
#  define HAVE_PETSC 1
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/distributed/tria.h>
namespace la {
template <>
struct Traits<TagPETSc>
{
    using TMTag    = TagPETSc;
    using Vector   = ::dealii::LinearAlgebraPETSc::MPI::BlockVector;
    using Matrix   = ::dealii::LinearAlgebraPETSc::MPI::BlockSparseMatrix;
    using IndexSet = ::dealii::IndexSet;
    
    using MatrixBlock   = ::dealii::LinearAlgebraPETSc::MPI::SparseMatrix;
    using VectorBlock   = ::dealii::LinearAlgebraPETSc::MPI::Vector;
    
    static constexpr bool IS_MPI = true;
};
}


#endif



// MPI mode with Trilinos
#ifdef DEAL_II_WITH_TRILINOS
#  define HAVE_TRILINOS 1
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/distributed/tria.h>
namespace la {
template <>
struct Traits<TagTrilinos>
{
    using TMTag    = TagTrilinos;
    using Vector   = ::dealii::LinearAlgebraTrilinos::MPI::BlockVector;
    using Matrix   = ::dealii::LinearAlgebraTrilinos::MPI::BlockSparseMatrix;
    using IndexSet = ::dealii::IndexSet;
    
    using MatrixBlock   = ::dealii::LinearAlgebraTrilinos::MPI::SparseMatrix;
    using VectorBlock   = ::dealii::LinearAlgebraTrilinos::MPI::Vector;
    
    static constexpr bool IS_MPI = true;
};
}

#endif




// add alias to simplify the code
# if defined(HAVE_TRILINOS) || defined(HAVE_PETSC)
#ifndef DISTRIBUTED_TRIA
#   define DISTRIBUTED_TRIA 1
template <int dim, int spacedim = dim>
using DTria = ::dealii::parallel::distributed::Triangulation<dim, spacedim>;
#   endif
# endif


// templated SolutionTransferSelector
template <int dim, typename VectorType, bool is_mpi, int spacedim=dim>
struct SolutionTransferSelector;


// specification for SolutionTransferSelector in serial mode
template <int dim, typename VectorType, int spacedim>
struct SolutionTransferSelector<dim, VectorType, /*is_mpi=*/false, spacedim>
{
    using type = dealii::SolutionTransfer<dim, VectorType, spacedim>;
};



// specification for SolutionTransferSelector in mpi mode
# if defined(HAVE_TRILINOS) || defined(HAVE_PETSC)
#include <deal.II/distributed/solution_transfer.h>
#   if !DEAL_II_VERSION_GTE(9, 7, 0)
template <int dim, typename VectorType, int spacedim>
struct SolutionTransferSelector<dim, VectorType, /*is_mpi=*/true, spacedim>
{
    using type = dealii::parallel::distributed::SolutionTransfer<
    dim, VectorType, spacedim>;
};

#   endif
# endif // #if defined(HAVE_TRILINOS) || defined(HAVE_PETSC)






#endif /* Traits_h */
