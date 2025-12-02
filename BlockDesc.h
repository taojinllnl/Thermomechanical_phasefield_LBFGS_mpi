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

struct Block {
    const unsigned int dim;
    const std::string  name;
    
    Block(const unsigned int dim,
          const std::string& name = "N/A")
    : dim(dim), name(name)
    {}
};

class BlockDesc
{
private:
    
    const std::vector<Block>                              __blocks;
 
    // the number of blocks
    const std::size_t                                    __nBlocks;
    
    // the index range for each block
    std::vector<std::array<unsigned int, 2>>              __dimRange;
    // the total number of dofs per node
 
    unsigned int                                          __nComponents;
    
    // tags for each component in a vector
    std::vector<unsigned int>                             __groupIDs;
    
    
    // dofs per block
    std::vector<dealii::types::global_dof_index>          __dofs_per_block;
public:
    
//    BlockDesc(const std::initializer_list<unsigned int> dims);
    
    BlockDesc(const std::initializer_list<unsigned int> dims,
              const std::initializer_list<std::string>  names = {});
    
    BlockDesc(const std::initializer_list<Block> blocks);
    
    const std::vector<std::array<unsigned int, 2>>& dimRange() const;
    const std::array<unsigned int, 2>& dimRange(unsigned int i) const;
    
    unsigned int nComponents() const;
    const std::vector<unsigned int>& componentTags() const;
    
    
    std::size_t nBlocks() const;
    
    const std::vector<dealii::types::global_dof_index>& dofsPerBlock() const;
    
    template <int dim, int spacedim = dim>
    void countDoFPerBlock(dealii::DoFHandler<dim, spacedim>& dof_handler)
    {
        // DoF Counting
        __dofs_per_block =
            dealii::DoFTools::count_dofs_per_fe_block(dof_handler, __groupIDs);
    }
    
};


class MPIBlockDesc : public BlockDesc
{
private:
    using IndexSet = dealii::IndexSet;
    std::vector<IndexSet>                   __owned_partitioning;
    std::vector<IndexSet>                   __relevant_partitioning;
    
    
    
public:
    MPIBlockDesc(const std::initializer_list<unsigned int> dims);
    
    const std::vector<IndexSet>& ownedPartitionint() const;
    const std::vector<IndexSet>& relevantPartitionint() const;
    
    
    template <int dim, int spacedim = dim>
    void update(dealii::DoFHandler<dim, spacedim>& dof_handler)
    {
        
        /*  *  *  *   *   *   *   *   *  MPI  *   *   *   *   *   *   *   *   */
        const IndexSet& locally_owned_dofs    = dof_handler.locally_owned_dofs();
        const IndexSet  locally_relevant_dofs =
            dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
        
        
        for(unsigned int i = 0; i < BlockDesc::nBlocks(); ++i)
        {
            const std::array<unsigned int, 2>& indices = dimRange(i);
            __owned_partitioning[i]    = locally_owned_dofs.get_view(indices[0], indices[1]);
            __relevant_partitioning[i] = locally_relevant_dofs.get_view(indices[0], indices[1]);
        }
        
        countDoFPerBlock(dof_handler);
    }
    
    
};
#endif /* BlockDesc_hpp */
