//
//  BlockVectorWrapper.cpp
//  main
//
//

#include "BlockVectorWrapper.h"

using namespace la;


template <typename TraitsType>
BlockVectorWrapper<TraitsType>
::BlockVectorWrapper(const BlockVectorWrapper& other)
: __hasRelevance(other.hasRelevance())
, __relevancePtr(__hasRelevance
                 ? std::make_unique<VecType>(*other.__relevancePtr)
                 : nullptr)
, __mpiInfo(other.__mpiInfo)
, __blockDesc(other.__blockDesc)
{}

template <typename TraitsType>
BlockVectorWrapper<TraitsType>
::BlockVectorWrapper(BlockVectorWrapper&& other) noexcept
: __hasRelevance(other.hasRelevance())
, __relevancePtr(std::move(other.__relevancePtr))
, __mpiInfo(other.__mpiInfo)
, __blockDesc(other.__blockDesc)
{}


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
        /*  *  *  *   *   *   *  serial version   *   *   *   *   *   *   *   */
        TraitsType::Vector::reinit(*__blockDesc.dofsPerBlock());
        /*  *  *  *   *   *   *  serial version   *   *   *   *   *   *   *   */
    } else {
        if(!__mpiInfo.isMPI())
        {
            return;
        }
        /*  *  *  *   *   *   *   *   *  MPI  *   *   *   *   *   *   *   *   */
        TraitsType::Vector::reinit(*(__blockDesc.ownedPartition()),
                                   *(__mpiInfo.mpiCommPtr()));
        TraitsType::Vector::operator=(0.0);
        
        if(__hasRelevance)
        {
            if (!__relevancePtr)
            {
                __relevancePtr = std::make_unique<VecType>();
            }
            
            __relevancePtr->reinit(*(__blockDesc.ownedPartition()),
                                   *(__blockDesc.relevantPartition()),
                                   *(__mpiInfo.mpiCommPtr()));
            
            (*__relevancePtr) = 0.0;
        }
        /*  *  *  *   *   *   *   *   *  MPI  *   *   *   *   *   *   *   *   */
        
    }
}


template <typename TraitsType>
BlockVectorWrapper<TraitsType>&
BlockVectorWrapper<TraitsType>
::operator= (const double s)
{
    TraitsType::Vector::operator=(s);
    return *this;
}


template <typename TraitsType>
BlockVectorWrapper<TraitsType>&
BlockVectorWrapper<TraitsType>
::operator= (const BlockVectorWrapper<TraitsType>& v)
{
    TraitsType::Vector::operator=(static_cast<const typename TraitsType::Vector&>(v));
    return *this;
}



template class la::BlockVectorWrapper<la::Traits<TagSerial>>;
template class la::BlockVectorWrapper<la::Traits<TagPETSc>>;
template class la::BlockVectorWrapper<la::Traits<TagTrilinos>>;
