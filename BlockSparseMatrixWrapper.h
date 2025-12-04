//
//  BlockSparseMatrixWrapper.hpp
//  main
//
//

#ifndef BlockSparseMatrixWrapper_hpp
#define BlockSparseMatrixWrapper_hpp

#include <functional>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_tools.h>



#include "MPIInfo.h"
#include "Traits.h"

#include "BlockDesc.h"

namespace la
{

template <typename TraitsType>
class BlockSparseMatrixWrapper
: public TraitsType::Matrix
{
public:
    using MatType       = typename TraitsType::Matrix;
    using CouplingItem  = dealii::DoFTools::Coupling;
    using Coupling      = dealii::Table<2, CouplingItem>;
    using CouplingFunc  = std::function<CouplingItem(const unsigned int,
                                                     const unsigned int)>;
    
    
    
private:
    static Coupling __couplingInit(const BlockDesc& blockDesc,
                                   const CouplingFunc& func);
    
    
    const MPIInfo&              __mpiInfo;
    const BlockDesc&            __blockDesc;
    const Coupling              __coupling;
    
    dealii::BlockSparsityPattern        __sparsity_pattern;
public:
    virtual ~BlockSparseMatrixWrapper() = default;
    
    BlockSparseMatrixWrapper() = delete;
    BlockSparseMatrixWrapper(const MPIInfo& mpiInfo,
                             const BlockDesc& blockDesc,
                             const CouplingFunc& func);
    
    
    template <int dim, int spacedim=dim>
    void init(dealii::DoFHandler<dim, spacedim>& dof_handler,
              const dealii::AffineConstraints<double>&  constraints = {},
              const bool keep_constrained_dofs = true,
              const dealii::types::subdomain_id subdomain_id = dealii::numbers::invalid_subdomain_id );
    
    
    BlockSparseMatrixWrapper& operator= (const BlockSparseMatrixWrapper&  m);
    BlockSparseMatrixWrapper& operator= (const double d);
};



template <typename TraitsType>
template <int dim, int spacedim>
void
BlockSparseMatrixWrapper<TraitsType>
::init(dealii::DoFHandler<dim, spacedim>&       dof_handler,
       const dealii::AffineConstraints<double>& constraints,
       const bool                               keep_constrained_dofs,
       const dealii::types::subdomain_id        subdomain_id)
{
    using namespace dealii;
    
    TraitsType::Matrix::clear();
    
    if constexpr (std::is_same_v<MatType, dealii::BlockSparseMatrix<double>>)
    {
        /*  *  *  *   *   *   *  serial version   *   *   *   *   *   *   *   */
        BlockDynamicSparsityPattern dsp(__blockDesc.dofsPerBlock(),
                                        __blockDesc.dofsPerBlock());
        
        DoFTools::make_sparsity_pattern(dof_handler,
                                        __coupling,
                                        dsp, 
                                        constraints,
                                        keep_constrained_dofs,
                                        subdomain_id);
        
        __sparsity_pattern.copy_from(dsp);

        TraitsType::Matrix::reinit(dsp);
        /*  *  *  *   *   *   *  serial version   *   *   *   *   *   *   *   */
    } else {
        if(!__mpiInfo.isMPI())
        {
            return;
        }
        
        /*  *  *  *   *   *   *   *   *  MPI  *   *   *   *   *   *   *   *   */
        const std::vector<IndexSet>& ownedPartition = *__blockDesc.ownedPartition();
        const std::vector<IndexSet>& relevPartition = *__blockDesc.relevantPartition();
        const IndexSet& locallyRelevantDoFs = *__blockDesc.localRelevantPartition();
        
        BlockDynamicSparsityPattern dsp(relevPartition);
        
        DoFTools::make_sparsity_pattern(dof_handler,
                                        __coupling,
                                        dsp, constraints,
                                        keep_constrained_dofs,
                                        subdomain_id);
        __sparsity_pattern.copy_from(dsp);
        
        
        SparsityTools::distribute_sparsity_pattern(dsp,
                                                   ownedPartition,
                                                   *__mpiInfo.mpiCommPtr(),
                                                   locallyRelevantDoFs);
        

        TraitsType::Matrix::reinit(ownedPartition,
                                   dsp,
                                   *__mpiInfo.mpiCommPtr());
        /*  *  *  *   *   *   *   *   *  MPI  *   *   *   *   *   *   *   *   */
    }
    

    
}




}

#endif /* BlockSparseMatrixWrapper_hpp */
