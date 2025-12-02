//
//  BlockDesc.cpp
//  main
//
//

#include "BlockDesc.h"


BlockDesc::BlockDesc(const std::initializer_list<unsigned int> dims,
                     const std::initializer_list<std::string>  names)
: _dims(dims)
, _nBlocks((unsigned int)_dims.size())
, __nComponents(0)
, __names(names)
{
    __dimRange.resize(_nBlocks);
    __componentTags.resize(0, 0);
        
    // if the length of the argument, names, is not fit to the number of blocks
    // "N/A" will be created for each block
    if (__names.size() != _nBlocks)
    {
        __names = std::vector<std::string>(_nBlocks, "N/A");
    }
    
    /// compute the block info
    unsigned int firstIndex = 0;
    unsigned int lastIndex  = firstIndex;
    for(unsigned int i = 0; i < _nBlocks; ++i)
    {
        lastIndex += _dims[i];
        __dimRange[i] = std::array<unsigned int, 2>{{firstIndex, lastIndex}};
        firstIndex = lastIndex;
        
        __componentTags.insert(__componentTags.end(), _dims[i], i);
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
    return __componentTags;
}

const std::vector<dealii::types::global_dof_index>& BlockDesc::dofsPerBlock() const
{
    return __dofs_per_block;
}
