//
//  CstPnt.h
//  main
//
//

#ifndef CstPnt_h
#define CstPnt_h

#include "CstSelectorBase.h"

namespace bcs
{
  template <typename Tria>
  class CstPnt : public CstSelectorBase<Tria>
  {
  public:
    static const int dim      = Tria::dimension;
    static const int spacedim = Tria::space_dimension;

  private:
    std::array<double, 3> pntCoorinates{};


  public:
    const double tol;

    virtual ~CstPnt() = default;

    CstPnt(const std::array<double, 3>       &point,
           const std::vector<CstEntry<Tria>> &csts,
           const double                       tol = 1e-9);

    CstPnt(const std::array<double, 3> &point,
           const CstEntry<Tria>        &csts,
           const double                 tol = 1e-9);

    CstPnt(const CstPnt &cstPnt);


    virtual std::size_t
    expectedNumberOfCstPoints() const override;
    virtual std::size_t
    expectedNumberOfCstEntries() const override;

    virtual bool
    isSelectedPnt(const ::dealii::Point<spacedim> &point) override;

    virtual std::unique_ptr<CstSelectorBase<Tria>>
    clone() const override;
  };
} // namespace bcs

#endif /* CstPnt_h */
