//
//  CstFunc.h
//  main
//
//

#ifndef CstFunc_h
#define CstFunc_h

#include "CstSelectorBase.h"

namespace bcs
{

  template <typename Tria>
  class CstFunc : public CstSelectorBase<Tria>
  {
  public:
    static const int dim      = Tria::dimension;
    static const int spacedim = Tria::space_dimension;

    using SelFunc =
      std::function<bool(const double, const double, const double)>;


    using ValuesAtPntFunc = std::function<
      void(const double, const double, const double, std::vector<double> &)>;

  private:
    std::shared_ptr<SelFunc>         selectorFunc{};
    std::shared_ptr<ValuesAtPntFunc> valuesFunc{};


    virtual void
    assignValues(const double         x,
                 const double         y,
                 const double         z,
                 std::vector<double> &values) override;

  public:
    virtual ~CstFunc() = default;

    CstFunc(const SelFunc                     &selectorFunc,
            const std::vector<CstEntry<Tria>> &csts);

    CstFunc(const SelFunc                     &selectorFunc,
            const std::vector<CstEntry<Tria>> &csts,
            const ValuesAtPntFunc             &valuesFunc);

    CstFunc(const CstFunc &cstFunc);



    virtual bool
    isSelectedPnt(const ::dealii::Point<spacedim> &point) override;
    virtual std::unique_ptr<CstSelectorBase<Tria>>
    clone() const override;
  };

} // namespace bcs

#endif /* CstFunc_h */
