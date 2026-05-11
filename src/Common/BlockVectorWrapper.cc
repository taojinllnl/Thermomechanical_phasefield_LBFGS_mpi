//
//  BlockVectorWrapper.cpp
//  main
//
//

#include "../../include/Common/BlockVectorWrapper.h"

using namespace ::dealii;
using namespace ::common;

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
        __relevancePtr->reinit(*(__blockDesc.ownedPartitionPtr()),
                               *(__blockDesc.relevantPartitionPtr()),
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
::initialize()
{
    
    if constexpr (std::is_same_v<VecType, dealii::BlockVector<double>>)
    {
        /*  *  *  *   *   *   *  serial version   *   *   *   *   *   *   *   */
        TraitsType::Vector::reinit(*__blockDesc.dofsPerBlockPtr());
        /*  *  *  *   *   *   *  serial version   *   *   *   *   *   *   *   */
    } else {
        if(!__mpiInfo.isMPI())
        {
            return;
        }
        /*  *  *  *   *   *   *   *   *  MPI  *   *   *   *   *   *   *   *   */
        TraitsType::Vector::reinit(*(__blockDesc.ownedPartitionPtr()),
                                   *(__mpiInfo.mpiCommPtr()));
        TraitsType::Vector::operator=(0.0);
        
        if(__hasRelevance)
        {
            __initRelevance();
            
            updateRelevance();
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
    TraitsType::Vector::operator=(v.base());
    if(v.__hasRelevance)
    {
        // only works for the relevance initialized in current object
        if (__relevancePtr) {
            (*__relevancePtr) = v.relevance();
        }
    }
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
::distributeCst(const dealii::AffineConstraints<double>& constraints,
                const bool updateGhostValues)
{
    constraints.distribute(base());
    
    if(updateGhostValues) updateRelevance();
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
        updateRelevance();
    } else {
        
        for(unsigned int i = 0; i < (*__blockDesc.dofsPerBlockPtr())[groupID]; ++i)
        {
            TraitsType::Vector::block(groupID)(i) = value;
        }
    }
}


template <typename TraitsType>
std::string BlockVectorWrapper<TraitsType>
::verificationInfo(const std::string vectorName)
{
    std::string content = "";
    
    content += "\n--------------------------------------------\n";
    content += vectorName + ": "+ std::to_string(++ithVer)  + "\n";
    content += "l1_norm: " + std::to_string(this->l1_norm()) + "\n";
    content += "l2_norm: " + std::to_string(this->l2_norm()) + "\n";
    content += "linfty_norm: " + std::to_string(this->linfty_norm()) + "\n";
    content += "mean: " + std::to_string(this->mean_value()) + "\n";
    content += "dot: " + std::to_string(base() * base()) + "\n";
    for (unsigned int ithGroup = 0; ithGroup < __blockDesc.nBlocks(); ++ithGroup)
    {
        content += "\t\tBlock " + std::to_string(ithGroup) + "\n";
        const auto& block = base().block(ithGroup);
        
        content += "\t\tl1_norm: " + std::to_string(block.l1_norm()) + "\n";
        content += "\t\tl2_norm: " + std::to_string(block.l2_norm()) + "\n";
        content += "\t\tlinfty_norm: " + std::to_string(block.linfty_norm()) + "\n";
        content += "\t\tmean: " + std::to_string(block.mean_value()) + "\n";
        content += "\t\tdot: " + std::to_string(block * block) + "\n";
    }
    if constexpr (std::is_same_v<typename TraitsType::TMTag, ::common::TagPETSc>) {
        if(__relevancePtr) {
            content += "__relevancePtr: \n";
            content += "l1_norm: " + std::to_string(__relevancePtr->l1_norm()) + "\n";
            content += "l2_norm: " + std::to_string(__relevancePtr->l2_norm()) + "\n";
            content += "linfty_norm: " + std::to_string(__relevancePtr->linfty_norm()) + "\n";
            content += "mean: " + std::to_string(__relevancePtr->mean_value()) + "\n";
            content += "dot: " + std::to_string((*__relevancePtr) * (*__relevancePtr)) + "\n";
            for (unsigned int ithGroup = 0; ithGroup < __blockDesc.nBlocks(); ++ithGroup)
            {
                content += "\t\tBlock " + std::to_string(ithGroup) + "\n";
                const auto& block = __relevancePtr->block(ithGroup);
                
                content += "\t\tl1_norm: " + std::to_string(block.l1_norm()) + "\n";
                content += "\t\tl2_norm: " + std::to_string(block.l2_norm()) + "\n";
                content += "\t\tlinfty_norm: " + std::to_string(block.linfty_norm()) + "\n";
                content += "\t\tmean: " + std::to_string(block.mean_value()) + "\n";
                content += "\t\tdot: " + std::to_string(block * block) + "\n";
            }
        }
    }
    content += "--------------------------------------------\n";
    return content;
}

template class ::common::BlockVectorWrapper<::common::Traits<::common::TagSerial>>;
#if defined(HAVE_PETSC) && HAVE_PETSC
template class ::common::BlockVectorWrapper<::common::Traits<::common::TagPETSc>>;
#endif
#if defined(HAVE_TRILINOS) && HAVE_TRILINOS
  template class ::common::BlockVectorWrapper<::common::Traits<::common::TagTrilinos>>;
#endif
