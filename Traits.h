//
//  Traits.h
//  main
//
//

#ifndef Traits_h
#define Traits_h


#include <deal.II/lac/generic_linear_algebra.h>


//
//namespace mpi
//{
//
//#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
//!(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
//using namespace dealii::LinearAlgebraPETSc;
//#  define USE_PETSC_LA
////#include <deal.II/lac/petsc_solver.h>
//
//#elif defined(DEAL_II_WITH_TRILINOS)
//using namespace dealii::LinearAlgebraTrilinos;
////#include <deal.II/lac/trilinos_solver.h>
//
//#else
//#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
//#endif
//
//
//}


namespace la {

struct TagSerial    {};
struct TagPETSc     {};
struct TagTrilinos  {};



// templated class for different scenarios for class types
template <typename BackendTag>
struct Traits;


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

template <int dim, int spacedim = dim>
using RTria = ::dealii::Triangulation<dim, spacedim>;





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


#ifndef DISTRIBUTED_TRIA
#   define DISTRIBUTED_TRIA 1
template <int dim, int spacedim = dim>
using DTria = ::dealii::parallel::distributed::Triangulation<dim, spacedim>;
#endif

#endif




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

#ifndef DISTRIBUTED_TRIA
#   define DISTRIBUTED_TRIA 1
template <int dim, int spacedim = dim>
using DTria = ::dealii::parallel::distributed::Triangulation<dim, spacedim>;
#endif


#endif



#endif /* Traits_h */
