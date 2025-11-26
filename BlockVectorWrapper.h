//
//  BlockVectorWrapper.h
//  main
//
//

#include <memory>

#include "Traits.h"

#ifndef BlockVectorWrapper_h
#define BlockVectorWrapper_h



namespace PhaseField_monolithic
{
namespace la
{

template <typename TraitsType>
class BlockVectorWrapper
: public TraitsType::Vector
{
private:
    mutable bool __hasRelevance;
    mutable std::unique_ptr<typename TraitsType::Vector> __relevancePtr{};
    
public:
    virtual ~BlockVectorWrapper() = default;
    
    BlockVectorWrapper() = delete;
    BlockVectorWrapper(const bool hasRelevance=true);
    
    
    const typename TraitsType::Vector& relevance() const;
    bool hasRelevance() const;
    
    
    void reinit();
    
};


template <typename TraitsType>
BlockVectorWrapper<TraitsType>
::BlockVectorWrapper(const bool hasRelevance)
: __hasRelevance(hasRelevance)
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
    
}



}
}


#endif /* BlockVectorWrapper_h */
