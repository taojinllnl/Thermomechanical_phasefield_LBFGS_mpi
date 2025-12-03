//
//  BlockVectorWrapper.cpp
//  main
//
//

#include "BlockVectorWrapper.h"

using namespace la;

template <typename TraitsType>
BlockVectorWrapper<TraitsType>
::BlockVectorWrapper(const MPIInfo& mpiInfo,
                     const BlockDesc& blockDesc,
                     const bool hasRelevance)
: __hasRelevance(hasRelevance)
, __relevancePtr(__hasRelevance? std::make_unique<VecType>() : nullptr)
, __mpiInfo(mpiInfo)
, __blockDesc(blockDesc)
{}


template <typename TraitsType>
const typename TraitsType::Vector&
BlockVectorWrapper<TraitsType>
::relevance() const
{
    if(!__hasRelevance)
    {
        __hasRelevance = true;
    }
    return *__relevancePtr;
}


template <typename TraitsType>
bool
BlockVectorWrapper<TraitsType>
::hasRelevance() const
{
    return __hasRelevance;
}


template <typename TraitsType>
void
BlockVectorWrapper<TraitsType>
::reinit()
{
    
    if constexpr (std::is_same_v<VecType, dealii::BlockVector<double>>)
    {
        // TODO: serial version
    } else {
        if(!__mpiInfo.isMPI())
        {
            return;
        }
        /*  *  *  *   *   *   *   *   *  MPI  *   *   *   *   *   *   *   *   */
        TraitsType::Vector::reinit(*(__blockDesc.ownedPartition()),
                                   *(__mpiInfo.mpiComm()));
        TraitsType::Vector::operator=(0.0);
        
        if(__hasRelevance)
        {
            if (!__relevancePtr)
            {
                __relevancePtr = std::make_unique<VecType>();
            }
            
            __relevancePtr->reinit(*(__blockDesc.ownedPartition()),
                                   *(__blockDesc.relevantPartition()),
                                   *(__mpiInfo.mpiComm()));
            
            (*__relevancePtr) = 0.0;
        }
        /*  *  *  *   *   *   *   *   *  MPI  *   *   *   *   *   *   *   *   */
        
    }
}




template class la::BlockVectorWrapper<la::Traits<TagSerial>>;
template class la::BlockVectorWrapper<la::Traits<TagPETSc>>;
template class la::BlockVectorWrapper<la::Traits<TagTrilinos>>;
