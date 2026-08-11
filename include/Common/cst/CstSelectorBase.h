//
//  CstSelectorBase.h
//  main
//
//

#ifndef CstSelectorBase_h
#define CstSelectorBase_h

#include "../MPIInfo.h"
#include "CstEntry.h"

namespace bcs
{

  // This class is designed to store the input and output data.
  template <typename Tria>
  class CstSelectorBase
  {
  public:
    static const int dim      = Tria::dimension;
    static const int spacedim = Tria::space_dimension;

    using GlobalDoF = typename CstEntryResult<Tria>::GlobalDoF;

    static constexpr const double INF = std::numeric_limits<double>::infinity();

    static constexpr std::size_t UNKNOW_SIZE =
      std::numeric_limits<std::size_t>::max();

    struct CstPointRecord
    {
      std::array<double, 3>    point{};
      std::vector<std::size_t> entryIndices{};
    };

  private:
    std::vector<CstEntry<Tria>>       __cstsInput;
    std::vector<CstEntryResult<Tria>> __cstsOutput;

    std::array<double, 3>  __pntCache;
    std::vector<double>    __valuesCache;
    std::vector<GlobalDoF> __dofsCache;

    std::vector<CstPointRecord> __cstPntRecordsFound;

    // the variable to guarantee no same dof been extracted
    std::set<GlobalDoF> __dofsFound;



    bool hasInit = false;


    void
    clearCache();
    void
    clearFound();


  protected:
    virtual void
    assignValues(const double         x,
                 const double         y,
                 const double         z,
                 std::vector<double> &values);



  public:
    virtual ~CstSelectorBase() = default;

    CstSelectorBase(const CstSelectorBase &selector);
    CstSelectorBase(CstSelectorBase &&selector);

    CstSelectorBase(const CstEntry<Tria> &cstInput);
    CstSelectorBase(const std::vector<CstEntry<Tria>> &cstInput);
    CstSelectorBase(const std::initializer_list<CstEntry<Tria>> &cstInput);


    static bool
    isDoFValid(const GlobalDoF dof);
    bool
    hasNoCachedDoF(const CstEntry<Tria> &entryRef) const;


    virtual std::size_t
    expectedNumberOfCstPoints() const;
    virtual std::size_t
    expectedNumberOfCstEntries() const;

    std::size_t
    nCstPnts() const;
    std::size_t
    nCstEntries() const;

    void
    newCstInput(const std::vector<CstEntry<Tria>> &cstInput);


    const std::vector<CstEntry<Tria>> &
    cstInput() const;
    const std::vector<CstEntryResult<Tria>> &
    cstOutput() const;

    const std::vector<CstPointRecord> &
    cstPointRecordsFound() const;

    // Clear all cache data without changing cstInput.
    void
    clear();

    virtual bool
    isSelectedPnt(const ::dealii::Point<spacedim> &point) = 0;

    virtual std::unique_ptr<CstSelectorBase<Tria>>
    clone() const = 0;

    void
    init(const std::size_t nDoFs);

    std::vector<GlobalDoF> &
    addDoFs2Cache(const std::size_t nDoFs);
    void
    addDoF2Cache(const std::size_t     nDoFs,
                 const GlobalDoF       dof,
                 const CstEntry<Tria> &entryRef);

    void
    createOutputByPoint(const ::dealii::Point<spacedim> &point,
                        const std::size_t                nDoFs);


    void
    updateValues(const std::size_t nDoFs);
  };
} // namespace bcs

#endif /* CstSelectorBase_h */
