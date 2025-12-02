//
//  BlockDesc.cpp
//  main
//
//

#include "BlockDesc.h"


BlockDesc::Block::Block(const unsigned int dim,
                        const std::string& name)
: dim(dim)
, name(name)
{}


unsigned int 
BlockDesc::__nComponentsInit(const std::vector<Block>& blocks)
{
    unsigned int sum = 0;
    for (const Block& b : blocks)
        sum += b.dim;
    return sum;
}

std::vector<std::array<unsigned int, 2>> 
BlockDesc::__dimRangeInit(const std::vector<Block>& blocks)
{
    std::vector<std::array<unsigned int, 2>> ranges;
    
    ranges.reserve(blocks.size());
    
    unsigned int firstIndex = 0;
    unsigned int lastIndex  = firstIndex;
    for (const Block& b : blocks)
    {
        lastIndex += b.dim;
        ranges.push_back({firstIndex, lastIndex});
        firstIndex = lastIndex;
    }
    return ranges;
}


std::vector<unsigned int> 
BlockDesc
::__groupIDsInit(const std::vector<Block>& blocks)
{
    std::vector<unsigned int> groupIDs;
    unsigned int id = 0;
    for (const Block& b : blocks)
    {
        groupIDs.insert(groupIDs.end(), b.dim, id++);
    }
    return groupIDs;
}


BlockDesc::BlockDesc(const MPIInfo&                     mpiInfo,
                     const std::initializer_list<Block> blocks)
: __mpiInfo(mpiInfo)
, __blocks(blocks)
, __nBlocks(__blocks.size())
, __dimRange(BlockDesc::__dimRangeInit(blocks))
, __nComponents(BlockDesc::__nComponentsInit(blocks))
, __groupIDs(BlockDesc::__groupIDsInit(blocks))
{}


const std::vector<std::array<unsigned int, 2>>& 
BlockDesc::dimRange() const
{
    return __dimRange;
}


const std::array<unsigned int, 2>& 
BlockDesc::dimRange(unsigned int ithGroup) const
{
    return __dimRange[ithGroup];
}

unsigned int BlockDesc::nComponents() const
{
    return __nComponents;
}

const std::vector<unsigned int>& BlockDesc::groupIDs() const
{
    return __groupIDs;
}

unsigned int BlockDesc::ithGroupID(const unsigned int ithComponent) const
{
    return __groupIDs[ithComponent];
}

std::size_t BlockDesc::nBlocks() const
{
    return __nBlocks;
}

const std::vector<dealii::types::global_dof_index>*
BlockDesc::dofsPerBlock() const
{
    if(!__dofs_per_block)
    {
        std::cout << "[ ERROR ] un-updated dofsPerBlock" << std::endl;
        return nullptr;
    }
    return __dofs_per_block.get();
}

const std::vector<BlockDesc::IndexSet>*
BlockDesc::ownedPartitionint() const
{
    if (!__mpiInfo.isMPI() || !__owned_partitioning) 
    {
        std::cout << "[ ERROR ] non-MPI mode or un-updated __owned_partitioning." << std::endl;
        return nullptr;
    }
    return __owned_partitioning.get();
}

const std::vector<BlockDesc::IndexSet>*
BlockDesc::relevantPartitionint() const
{
    if (!__mpiInfo.isMPI() || !__relevant_partitioning) 
    {
        std::cout << "[ ERROR ] non-MPI mode or un-updated __relevant_partitioning." << std::endl;
        return nullptr;
    }
    return __relevant_partitioning.get();
}





void BlockDesc::summary(std::ostream& stream)
{
    stream << __nBlocks << " block(s) with " << __nComponents << " components: " << std::endl;
    
    for (unsigned int i = 0; i < __nBlocks; ++i)
    {
        stream << "\tBlock No. " << i << ":\n\t\tdim: " << __blocks[i].dim
        << "\tname: " << __blocks[i].name;
        stream << "\n\t\tdim range: [ " << __dimRange[i][0] <<  ", " << __dimRange[i][1] << " ) " << std::endl;
    }
    
    stream << "Group IDs :  ";
    for (const unsigned int groupID : __groupIDs) 
    {
        stream << groupID << " ";
    }
    stream << std::endl;
    
}



