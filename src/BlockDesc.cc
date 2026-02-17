//
//  BlockDesc.cpp
//  main
//
//

#include "../include/BlockDesc.h"


BlockDesc::Block::Block(const unsigned int dim,
                        const unsigned int groupID,
                        const std::string& name)
: dim(dim)
, name(name)
, groupID(groupID)
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


std::vector<BlockDesc::Block>
BlockDesc
::__blockInit(const std::initializer_list<InitType>& blocks)
{
    std::vector<Block> blockList;

    unsigned int i = 0;
    
    for(const InitType& init : blocks)
    {
        blockList.emplace_back(init.first, i, init.second);
        
        ++i;
    }
    
    return blockList;
}


BlockDesc::BlockDesc(const MPIInfo&                         mpiInfo,
                     const std::initializer_list<InitType>  blocks)
: __mpiInfo(mpiInfo)
, __blocks(BlockDesc::__blockInit(blocks))
, __nBlocks(__blocks.size())
, __dimRange(BlockDesc::__dimRangeInit(__blocks))
, __nComponents(BlockDesc::__nComponentsInit(__blocks))
, __groupIDs(BlockDesc::__groupIDsInit(__blocks))
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


std::size_t BlockDesc::nGroups() const
{
    return __groupIDs.size();
}

const std::vector<unsigned int>& BlockDesc::groupIDs() const
{
    return __groupIDs;
}

unsigned int BlockDesc::ithGroupID(const unsigned int ithComponent) const
{
    return __groupIDs[ithComponent];
}

unsigned int BlockDesc::ithGroupID(const std::string& name) const
{
    for (const Block& block : __blocks)
    {
        if (block.name == name) {
            return block.groupID;
        }
    }
    // TODO: ERROR
    return 0;
}

std::size_t BlockDesc::nBlocks() const
{
    return __nBlocks;
}

const std::vector<dealii::types::global_dof_index>*
BlockDesc::dofsPerBlockPtr() const
{
    if(!__dofs_per_block)
    {
        std::cout << "[ ERROR ] un-updated dofsPerBlock" << std::endl;
        return nullptr;
    }
    return __dofs_per_block.get();
}

const std::vector<BlockDesc::IndexSet>*
BlockDesc::ownedPartitionPtr() const
{
    if (!__mpiInfo.isMPI() || !__owned_partitioning) 
    {
        std::cout << "[ ERROR ] non-MPI mode or un-updated __owned_partitioning." << std::endl;
        return nullptr;
    }
    return __owned_partitioning.get();
}

const std::vector<BlockDesc::IndexSet>*
BlockDesc::relevantPartitionPtr() const
{
    if (!__mpiInfo.isMPI() || !__relevant_partitioning) 
    {
        std::cout << "[ ERROR ] non-MPI mode or un-updated __relevant_partitioning." << std::endl;
        return nullptr;
    }
    return __relevant_partitioning.get();
}

const BlockDesc::IndexSet*
BlockDesc::localRelevantPartition() const
{
    if (!__mpiInfo.isMPI() || !__localRelevantDoFs)
    {
        std::cout << "[ ERROR ] non-MPI mode or un-updated __localRelevantDoFs." << std::endl;
        return nullptr;
    }
    return __localRelevantDoFs.get();
}



void BlockDesc::summary(std::ostream& stream)
{
    stream << __nBlocks << " block(s) with " << __nComponents << " components: " << std::endl;
    
    for (unsigned int i = 0; i < __nBlocks; ++i)
    {
        stream << "\tBlock No. " << i << ":\n\t\tdim: " << __blocks[i].dim
        << "\tname: " << __blocks[i].name << "\tgroup ID: " << __blocks[i].groupID;
        stream << "\n\t\tdim range: [ " << __dimRange[i][0] <<  ", " << __dimRange[i][1] << " ) " << std::endl;
    }
    
    stream << "Group IDs :  ";
    for (const unsigned int groupID : __groupIDs) 
    {
        stream << groupID << " ";
    }
    stream << std::endl;
    
}



