//
//  BlockDesc.hpp
//  main
//
//

#ifndef BlockDesc_hpp
#define BlockDesc_hpp

#include <vector>
#include <string>
#include <initializer_list>
#include <ostream>

#include <memory>
#include "MPIInfo.h"

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

/// example:
///     _dims               = {3, 2, 1}
///     _nBlocks            = 3
///     __dimRange          = { [0, 3], [3, 5], [5, 6] }
///     __nComponents       = 6
///     __groupIDs          = {0, 0, 0, 1, 1, 2 }
///     __names             = { "a", "b", "c" }
///



class BlockDesc
{
public:
    struct Block {
        const unsigned int dim;
        const std::string  name;
        
        Block(const unsigned int dim,
              const std::string& name = "N/A");
    };
    
    
private:
    static unsigned int __nComponentsInit(const std::vector<Block>& blocks);

    static std::vector<std::array<unsigned int, 2>> 
    __dimRangeInit(const std::vector<Block>& blocks);
    
    static std::vector<unsigned int> __groupIDsInit(const std::vector<Block>& blocks);

private:
    using IndexSet = dealii::IndexSet;
    
    
    const MPIInfo&                                        __mpiInfo;
    
    const std::vector<Block>                              __blocks;
 
    // the number of blocks
    const std::size_t                                    __nBlocks;
    
    // the index range for each block
    const std::vector<std::array<unsigned int, 2>>       __dimRange;
    
    // the total number of dofs per node
    const unsigned int                                   __nComponents;
    
    // tags for each component in a vector
    const std::vector<unsigned int>                       __groupIDs;
    
    // dofs per block
    std::unique_ptr<std::vector<dealii::types::global_dof_index>> __dofs_per_block{};
    
    
    std::unique_ptr<std::vector<IndexSet>>              __owned_partitioning{};
    std::unique_ptr<std::vector<IndexSet>>              __relevant_partitioning{};
    
public:
    
    BlockDesc(const MPIInfo&                     mpiInfo,
              const std::initializer_list<Block> blocks);
    
    const std::vector<std::array<unsigned int, 2>>& dimRange() const;
    const std::array<unsigned int, 2>& dimRange(unsigned int ithGroup) const;
    
    std::size_t nBlocks() const;
    unsigned int nComponents() const;
    
    
    const std::vector<unsigned int>& groupIDs() const;
    unsigned int ithGroupID(const unsigned int ithComponent) const;

    
    template <int dim, int spacedim=dim>
    void updateDoFsInfo(dealii::DoFHandler<dim, spacedim>& dof_handler);
    
    const std::vector<dealii::types::global_dof_index>* dofsPerBlock() const;
    
    const std::vector<IndexSet>* ownedPartitionint() const;
    const std::vector<IndexSet>* relevantPartitionint() const;
    
    void summary(std::ostream& stream);
};


template <int dim, int spacedim>
void BlockDesc::updateDoFsInfo(dealii::DoFHandler<dim, spacedim>& dof_handler)
{
    if(!__dofs_per_block)
    {
        __dofs_per_block = std::make_unique<std::vector<dealii::types::global_dof_index>>();
    }
    
    if (__mpiInfo.isMPI())
    {
        if (!__owned_partitioning)
        {
            __owned_partitioning =
                std::make_unique<std::vector<IndexSet>>(__nBlocks);
        }
        else if (__owned_partitioning->size() != __nBlocks)
        {
            __owned_partitioning->assign(__nBlocks, IndexSet());
        }

        if (!__relevant_partitioning)
        {
            __relevant_partitioning =
                std::make_unique<std::vector<IndexSet>>(__nBlocks);
        }
        else if (__relevant_partitioning->size() != __nBlocks)
        {
            __relevant_partitioning->assign(__nBlocks, IndexSet());
        }

        
        /*  *  *  *   *   *   *   *   *  MPI  *   *   *   *   *   *   *   *   */
        const IndexSet& locally_owned_dofs    = dof_handler.locally_owned_dofs();
        const IndexSet  locally_relevant_dofs =
            dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
        
        
        for(unsigned int i = 0; i < BlockDesc::nBlocks(); ++i)
        {
            const std::array<unsigned int, 2>& indices = dimRange(i);
            (*__owned_partitioning)[i]    = locally_owned_dofs.get_view(indices[0], indices[1]);
            (*__relevant_partitioning)[i] = locally_relevant_dofs.get_view(indices[0], indices[1]);
        }
    }
    
    (*__dofs_per_block) =
        dealii::DoFTools::count_dofs_per_fe_block(dof_handler, __groupIDs);
}



#endif /* BlockDesc_hpp */
