//
//  BlockVectorWrapper.cpp
//  main
//
//

#include "BlockVectorWrapper.h"

using namespace la;
using namespace dealii;

template <typename TraitsType>
BlockVectorWrapper<TraitsType>
::BlockVectorWrapper(const BlockVectorWrapper& other)
: TraitsType::Vector(other.base())
, __hasRelevance(other.hasRelevance())
, __relevancePtr(__hasRelevance
                 ? std::make_unique<VecType>(other.relevance())
                 : nullptr)
, __mpiInfo(other.__mpiInfo)
, __blockDesc(other.__blockDesc)
{}

template <typename TraitsType>
BlockVectorWrapper<TraitsType>
::BlockVectorWrapper(BlockVectorWrapper&& other) noexcept
: TraitsType::Vector(other.base())
, __hasRelevance(other.hasRelevance())
, __relevancePtr(__hasRelevance 
                 ? std::move(other.__relevancePtr)
                 : nullptr)
, __mpiInfo(other.__mpiInfo)
, __blockDesc(other.__blockDesc)
{}


template <typename TraitsType>
BlockVectorWrapper<TraitsType>
::BlockVectorWrapper(const MPIInfo& mpiInfo,
                     const BlockDesc& blockDesc,
                     const bool hasRelevance)
: TraitsType::Vector()
, __hasRelevance(mpiInfo.isMPI() ? hasRelevance : false)
, __relevancePtr(__hasRelevance? std::make_unique<VecType>() : nullptr)
, __mpiInfo(mpiInfo)
, __blockDesc(blockDesc)
{}



template <typename TraitsType>
void
BlockVectorWrapper<TraitsType>
::__initRelevance()
{
    if (!__relevancePtr)
    {
        __relevancePtr = std::make_unique<VecType>();
    }
    
    if constexpr (!std::is_same_v<VecType, dealii::BlockVector<double>>)
        __relevancePtr->reinit(*(__blockDesc.ownedPartition()),
                               *(__blockDesc.relevantPartition()),
                               *(__mpiInfo.mpiCommPtr()));
}


template <typename TraitsType>
const typename TraitsType::Vector&
BlockVectorWrapper<TraitsType>
::updateRelevance()
{
    if (__hasRelevance) {
        
        // the relevant vector should already have the same dofs structure
        
        (*__relevancePtr) = base();
        __relevancePtr->update_ghost_values();
        
        return *__relevancePtr;
    } else {
        return base();
    }
    
}

template <typename TraitsType>
const typename TraitsType::Vector&
BlockVectorWrapper<TraitsType>
::relevance() const
{
    if (__hasRelevance) {
        return *__relevancePtr;
    } else {
        return base();
    }
    
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
::initalize()
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
            __initRelevance();
            
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


template <typename TraitsType>
typename TraitsType::Vector& 
BlockVectorWrapper<TraitsType>
::base()
{
    return *this;
}

template <typename TraitsType>
const typename TraitsType::Vector& 
BlockVectorWrapper<TraitsType>
::base() const
{
    return *this;
}



template <typename TraitsType>
void
BlockVectorWrapper<TraitsType>
::assignDoubleOverABlock(const unsigned int groupID,
                         const double value)
{
    if constexpr (is_mpi)
    {
        auto &vecBlock = TraitsType::Vector::block(groupID);
        const IndexSet &owned = vecBlock.locally_owned_elements();
        
        for (auto it = owned.begin(); it != owned.end(); ++it)
            vecBlock[*it] = value;
        
        TraitsType::Vector::compress(VectorOperation::insert);
    } else {
        
        for(unsigned int i = 0; i < (*__blockDesc.dofsPerBlock())[groupID]; ++i)
        {
            TraitsType::Vector::block(groupID)(i) = value;
        }
    }
}



template class la::BlockVectorWrapper<la::Traits<TagSerial>>;
template class la::BlockVectorWrapper<la::Traits<TagPETSc>>;
template class la::BlockVectorWrapper<la::Traits<TagTrilinos>>;
