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
    
    const MPIInfo&      __mpiInfo;
    const BlockDesc&    __blockDesc;
public:
    virtual ~BlockVectorWrapper() = default;
    
    BlockVectorWrapper() = delete;
    
    BlockVectorWrapper(const BlockVectorWrapper& other);
    BlockVectorWrapper(BlockVectorWrapper&& other) noexcept;
    
    BlockVectorWrapper(const MPIInfo& mpiInfo,
                       const BlockDesc& blockDesc,
                       const bool hasRelevance=true);
    
    
    const typename TraitsType::Vector& relevance() const;
    bool hasRelevance() const;
    
    
    void initalize();
    
    BlockVectorWrapper&     operator= (const double s);
    BlockVectorWrapper&     operator= (const BlockVectorWrapper& v);
    
    
    typename TraitsType::Vector& base();
    const typename TraitsType::Vector& base() const;
};





}



#endif /* BlockVectorWrapper_h */
