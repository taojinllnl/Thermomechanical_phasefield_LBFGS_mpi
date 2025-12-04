//
//  BlockSparseMatrixWrapper.cpp
//  main
//
//

#include "BlockSparseMatrixWrapper.h"
using namespace la;

template <typename TraitsType>
using BSMatrix = BlockSparseMatrixWrapper<TraitsType>;


template <typename TraitsType>
typename BSMatrix<TraitsType>::Coupling
BlockSparseMatrixWrapper<TraitsType>
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
BlockSparseMatrixWrapper<TraitsType>
::BlockSparseMatrixWrapper(const MPIInfo& mpiInfo,
                             const BlockDesc& blockDesc,
                             const BSMatrix<TraitsType>::CouplingFunc& func)
: __mpiInfo(mpiInfo)
, __blockDesc(blockDesc)
, __coupling(BSMatrix<TraitsType>::__couplingInit(blockDesc, func))
{}


template <typename TraitsType>
BlockSparseMatrixWrapper<TraitsType>&
BlockSparseMatrixWrapper<TraitsType>
::operator= (const BlockSparseMatrixWrapper<TraitsType>&  m)
{
    TraitsType::Matrix::operator=(static_cast<const typename TraitsType::Matrix&>(m));
    return *this;
}


template <typename TraitsType>
BlockSparseMatrixWrapper<TraitsType>&
BlockSparseMatrixWrapper<TraitsType>
::operator= (const double d)
{
    TraitsType::Matrix::operator=(d);
    return *this;
}



template class la::BlockSparseMatrixWrapper<la::Traits<TagSerial>>;
template class la::BlockSparseMatrixWrapper<la::Traits<TagPETSc>>;
template class la::BlockSparseMatrixWrapper<la::Traits<TagTrilinos>>;
