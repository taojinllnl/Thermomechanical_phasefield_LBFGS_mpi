//
//  BlockVectorWrapper.h
//  main
//
//


#include <deal.II/lac/affine_constraints.h>

#include <deal.II/dofs/dof_handler.h>

#include <memory>

#include "Traits.h"

#include "MPIInfo.h"

#include "BlockDesc.h"



#ifndef BlockVectorWrapper_h
#define BlockVectorWrapper_h



namespace la
{


/**
 *
 * This class is a light-weight wrapper for deal.II BlockVector type ( defined in `TraitsType::Vector`), providing convenience operations in both serial and distributed (MPI) executions.
 *
 * This wrapper derives from the underlying vector type so it can be used in the same way as deal.II vectors.
 * When an API requires an exact reference to the underlying type, `base()` can be used to access real type of the vector.
 * The locally owned (non-ghosted) vector will be returned in mpi mode.
 *
 * In MPI mode, the wrapper can optionally maintain a locally relevant (ghosted) vector to simplify ghost value updates.
 * The ghosted vector is accessed via `relevance()` and gets uppated by `updateRelevance()`.
 *
 * Notes:
 * 1. `initialize()` must be called after construction to (re)initialize internal storage for the one with locally relevant (ghosted) vector.
 * 2. `base()` always returns the locally owned (non-ghosted) vector.
 *
 */

template <typename TraitsType>
class BlockVectorWrapper
: public TraitsType::Vector
{
public:
    using VecType = typename TraitsType::Vector;
    
    static constexpr bool is_mpi = TraitsType::IS_MPI;
private:
    
    const bool __hasRelevance;
    mutable std::unique_ptr<VecType> __relevancePtr{};
    
    const MPIInfo&      __mpiInfo;
    const BlockDesc&    __blockDesc;
    
    
    void __initRelevance();
    
    
public:
    virtual ~BlockVectorWrapper() = default;
    
    BlockVectorWrapper() = delete;
    
    BlockVectorWrapper(const BlockVectorWrapper& other);
    BlockVectorWrapper(BlockVectorWrapper&& other) noexcept;
    
    BlockVectorWrapper(const MPIInfo& mpiInfo,
                       const BlockDesc& blockDesc,
                       const bool hasRelevance=false);
    
    const typename TraitsType::Vector& updateRelevance();
    const typename TraitsType::Vector& relevance() const;
    bool hasRelevance() const;
    
    void initialize();
    
    BlockVectorWrapper&     operator= (const double s);
    BlockVectorWrapper&     operator= (const BlockVectorWrapper& v);
    
    
    typename TraitsType::Vector& base();
    const typename TraitsType::Vector& base() const;
    
    
    void assignDoubleOverABlock(const unsigned int groupID,
                                const double value);
    
    void distributeCst(const dealii::AffineConstraints<double>& constraints,
                       const bool updateGhostValues = true);
    unsigned int ithVer = 0;
    std::string verificationInfo(const std::string vectorName);
    
    template <int dim, int spacedim = dim>
    void copyNoncst(const BlockVectorWrapper& other,
                    const dealii::AffineConstraints<double>& constraints,
                    const dealii::DoFHandler<dim, spacedim>& dof_handler);
    
    template <int dim, int spacedim = dim>
    void copyAndRemoveCst(const BlockVectorWrapper& other,
                          const dealii::AffineConstraints<double>& constraints,
                          const dealii::DoFHandler<dim, spacedim>& dof_handler);
};



template <typename TraitsType>
template <int dim, int spacedim>
void
BlockVectorWrapper<TraitsType>
::copyNoncst(const BlockVectorWrapper& other,
             const dealii::AffineConstraints<double>& constraints,
             const dealii::DoFHandler<dim, spacedim>& dof_handler)
{
    if constexpr (is_mpi) {
        
        const dealii::IndexSet &owned = dof_handler.locally_owned_dofs();
        
        for (auto i = owned.begin(); i != owned.end(); ++i)
        {
            const dealii::types::global_dof_index dof = *i;
            if (!constraints.is_constrained(dof))
                base()(dof) = other(dof);
            else
                base()(dof) = 0.0;
        }
        
        base().compress(dealii::VectorOperation::insert);
        updateRelevance();
    } else {
        
        for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
            if (!constraints.is_constrained(i))
                base()(i) = other(i);
    }
}



template <typename TraitsType>
template <int dim, int spacedim>
void
BlockVectorWrapper<TraitsType>
::copyAndRemoveCst(const BlockVectorWrapper& other,
                   const dealii::AffineConstraints<double>& constraints,
                   const dealii::DoFHandler<dim, spacedim>& dof_handler)
{
    base() = other.base();
    
    if constexpr (is_mpi) {
        
        const dealii::IndexSet &owned = dof_handler.locally_owned_dofs();
        
        for (auto i = owned.begin(); i != owned.end(); ++i)
        {
            const dealii::types::global_dof_index dof = *i;
            if (constraints.is_constrained(dof))
                base()(dof) = 0.0;
        }
        
        base().compress(dealii::VectorOperation::insert);
        updateRelevance();
    } else {
        
        for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
            if (constraints.is_constrained(i))
                base()(i) = 0.0;
    }
}






}



#endif /* BlockVectorWrapper_h */
