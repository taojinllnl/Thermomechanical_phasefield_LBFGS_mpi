//
//  BlockVectorWrapper.h
//  main
//
//

#include <memory>

#include "Traits.h"

#include "MPIInfo.h"

#include "BlockDesc.h"

#ifndef BlockVectorWrapper_h
#define BlockVectorWrapper_h



namespace la
{

template <typename TraitsType>
class BlockVectorWrapper
: public TraitsType::Vector
{
private:
    using VecType = typename TraitsType::Vector;
    mutable bool __hasRelevance;
    mutable std::unique_ptr<VecType> __relevancePtr{};
    
    const MPIInfo& __mpiInfo;
public:
    virtual ~BlockVectorWrapper() = default;
    
    BlockVectorWrapper() = delete;
    BlockVectorWrapper(const MPIInfo& mpiInfo, const bool hasRelevance=true);
    
    
    const typename TraitsType::Vector& relevance() const;
    bool hasRelevance() const;
    
    
    void reinit(const BlockDesc& blockDesc);
    
};





}



#endif /* BlockVectorWrapper_h */
