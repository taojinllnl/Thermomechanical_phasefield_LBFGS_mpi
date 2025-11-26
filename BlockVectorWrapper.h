//
//  BlockVectorWrapper.h
//  main
//
//

#include "Traits.h"

#ifndef BlockVectorWrapper_h
#define BlockVectorWrapper_h



namespace PhaseField_monolithic
{
namespace la
{

template <typename BackendTag>
class BlockVectorWrapper
: public Traits<BackendTag>::Vector
{
private:
    mutable bool __hasRelevance;
    mutable typename Traits<BackendTag>::Vector __relevance;
    
public:
    virtual ~BlockVectorWrapper() = default;
    
    BlockVectorWrapper() = delete;
    BlockVectorWrapper(const bool hasRelevance=true);
    
    
    const typename Traits<BackendTag>::Vector& relevance() const;
    bool hasRelevance() const;
    
    
    void reinit();
    
};


template <typename BackendTag>
BlockVectorWrapper<BackendTag>
::BlockVectorWrapper(const bool hasRelevance)
: __hasRelevance(hasRelevance)
{}


template <typename BackendTag>
const typename Traits<BackendTag>::Vector&
BlockVectorWrapper<BackendTag>
::relevance() const
{
    if(!__hasRelevance)
    {
        __hasRelevance = true;
    }
    return __relevance;
}


template <typename BackendTag>
bool
BlockVectorWrapper<BackendTag>
::hasRelevance() const
{
    return __hasRelevance;
}


template <typename BackendTag>
void 
BlockVectorWrapper<BackendTag>
::reinit()
{
    
}



}
}


#endif /* BlockVectorWrapper_h */
