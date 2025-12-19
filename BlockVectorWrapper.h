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
public:
    using VecType = typename TraitsType::Vector;
    
    static constexpr bool is_mpi = TraitsType::IS_MPI;
private:
    
    const bool __hasRelevance;
    mutable std::unique_ptr<VecType> __relevancePtr{};
    
    const MPIInfo&      __mpiInfo;
    const BlockDesc&    __blockDesc;
    
    
    void __initRelevance();
    
    
public:
    virtual ~BlockVectorWrapper() = default;
    
    BlockVectorWrapper() = delete;
    
    BlockVectorWrapper(const BlockVectorWrapper& other);
    BlockVectorWrapper(BlockVectorWrapper&& other) noexcept;
    
    BlockVectorWrapper(const MPIInfo& mpiInfo,
                       const BlockDesc& blockDesc,
                       const bool hasRelevance=false);
    
    const typename TraitsType::Vector& updateRelevance();
    const typename TraitsType::Vector& relevance() const;
    bool hasRelevance() const;
    
    void initalize();
    
    BlockVectorWrapper&     operator= (const double s);
    BlockVectorWrapper&     operator= (const BlockVectorWrapper& v);
    
    
    typename TraitsType::Vector& base();
    const typename TraitsType::Vector& base() const;
    
    
    void assignDoubleOverABlock(const unsigned int groupID,
                                const double value);

};





}



#endif /* BlockVectorWrapper_h */
