//
//  BlockDesc.cpp
//  main
//
//

#include "BlockDesc.h"



BlockDesc::BlockDesc(const std::initializer_list<Block> blocks)
: __blocks(blocks)
, __nBlocks(__blocks.size())
, __nComponents(0)
{
    __dimRange.resize(__nBlocks);
    __groupIDs.resize(0, 0);

    
    /// compute the block info
    unsigned int firstIndex = 0;
    unsigned int lastIndex  = firstIndex;
    for(unsigned int i = 0; i < __nBlocks; ++i)
    {
        lastIndex += __blocks[i].dim;
        __dimRange[i] = std::array<unsigned int, 2>{{firstIndex, lastIndex}};
        firstIndex = lastIndex;
        
        __groupIDs.insert(__groupIDs.end(), __blocks[i].dim, i);
    }
    __nComponents = lastIndex;
}


const std::vector<std::array<unsigned int, 2>>& BlockDesc::dimRange() const
{
    return __dimRange;
}


const std::array<unsigned int, 2>& BlockDesc::dimRange(unsigned int i) const
{
    return __dimRange[i];
}

unsigned int BlockDesc::nComponents() const
{
    return __nComponents;
}

const std::vector<unsigned int>& BlockDesc::componentTags() const
{
    return __groupIDs;
}

std::size_t BlockDesc::nBlocks() const
{
    return __nBlocks;
}

const std::vector<dealii::types::global_dof_index>& BlockDesc::dofsPerBlock() const
{
    return __dofs_per_block;
}
