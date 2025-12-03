//
//  BlockSparsityMatrixWrapper.cpp
//  main
//
//

#include "BlockSparsityMatrixWrapper.h"
using namespace la;

template <typename TraitsType>
using BSMatrix = BlockSparsityMatrixWrapper<TraitsType>;


template <typename TraitsType>
typename BSMatrix<TraitsType>::Coupling
BlockSparsityMatrixWrapper<TraitsType>
::__couplingInit(const BlockDesc& blockDesc,
                 const BSMatrix<TraitsType>::CouplingFunc& func)
{
    const unsigned int nComponents = blockDesc.nComponents();
    
    Coupling coupling(nComponents, nComponents);
    
    for (unsigned int ii = 0; ii < nComponents; ++ii)
        for (unsigned int jj = 0; jj < nComponents; ++jj)
            coupling[ii][jj] = func(ii, jj);
            
    return coupling;
}



template <typename TraitsType>
BlockSparsityMatrixWrapper<TraitsType>
::BlockSparsityMatrixWrapper(const MPIInfo& mpiInfo,
                             const BlockDesc& blockDesc,
                             const BSMatrix<TraitsType>::CouplingFunc& func)
: __mpiInfo(mpiInfo)
, __blockDesc(blockDesc)
, __coupling(BSMatrix<TraitsType>::__couplingInit(blockDesc, func))
{}





template class la::BlockSparsityMatrixWrapper<la::Traits<TagSerial>>;
template class la::BlockSparsityMatrixWrapper<la::Traits<TagPETSc>>;
template class la::BlockSparsityMatrixWrapper<la::Traits<TagTrilinos>>;
