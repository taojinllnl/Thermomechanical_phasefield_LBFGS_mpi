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

template <typename TraitsType>
typename TraitsType::Matrix& 
BlockSparseMatrixWrapper<TraitsType>
::base()
{
    return *this;
}

template <typename TraitsType>
const typename TraitsType::Matrix& 
BlockSparseMatrixWrapper<TraitsType>
::base() const
{
    return *this;
}



template <typename TraitsType>
std::string BlockSparseMatrixWrapper<TraitsType>
::verificationInfo()
{
    std::string content = "";
    
    content += "\n--------------------------------------------\n";
    content += std::to_string(++ithVer) + "\n";
    content += "frobenius_norm: " + std::to_string(this->frobenius_norm()) + "\n";
    for (unsigned int ithGroup = 0; ithGroup < __blockDesc.nBlocks(); ++ithGroup)
        for (unsigned int jthGroup = 0; jthGroup < __blockDesc.nBlocks(); ++jthGroup)
        {
            content += "\t\tBlock " + std::to_string(ithGroup) + ", " +  std::to_string(jthGroup) +  "\n";
            const auto& block = base().block(ithGroup, jthGroup);
            
            content += "\t\tfrobenius_norm: " + std::to_string(block.frobenius_norm()) + "\n";
        }
    
    content += "--------------------------------------------\n";
    return content;
}


template class la::BlockSparseMatrixWrapper<la::Traits<TagSerial>>;
template class la::BlockSparseMatrixWrapper<la::Traits<TagPETSc>>;
template class la::BlockSparseMatrixWrapper<la::Traits<TagTrilinos>>;
