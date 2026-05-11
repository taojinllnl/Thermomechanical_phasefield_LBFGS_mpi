//
//  CstFunc.cpp
//  main
//
//



#include "../../../include/Common/cst/CstFunc.h"

using namespace ::dealii;
using namespace ::bcs;



template <typename Tria>
CstFunc<Tria>::CstFunc(const SelFunc                     &selectorFunc,
                       const std::vector<CstEntry<Tria>> &csts)
  : CstSelectorBase<Tria>(csts)
  , selectorFunc(std::make_shared<SelFunc>(selectorFunc))
  , valuesFunc(nullptr)
{}


template <typename Tria>
CstFunc<Tria>::CstFunc(const SelFunc                     &selectorFunc,
                       const std::vector<CstEntry<Tria>> &csts,
                       const ValuesAtPntFunc             &valuesFunc)
  : CstSelectorBase<Tria>(csts)
  , selectorFunc(std::make_shared<SelFunc>(selectorFunc))
  , valuesFunc(std::make_shared<ValuesAtPntFunc>(valuesFunc))
{}


template <typename Tria>
CstFunc<Tria>::CstFunc(const CstFunc &cstFunc)
  : CstSelectorBase<Tria>(cstFunc)
  , selectorFunc(cstFunc.selectorFunc)
  , valuesFunc(cstFunc.valuesFunc)
{}



template <typename Tria>
bool
CstFunc<Tria>::isSelectedPnt(const ::dealii::Point<spacedim> &point)
{
  bool isCurrentPoint = false;

  if (selectorFunc)
    {
      const auto &func = *selectorFunc;
      if constexpr (spacedim == 1)
        {
          isCurrentPoint = func(point[0], 0.0, 0.0);
        }
      else if constexpr (spacedim == 2)
        {
          isCurrentPoint = func(point[0], point[1], 0);
        }
      else if constexpr (spacedim == 3)
        {
          isCurrentPoint = func(point[0], point[1], point[2]);
        }
    }

  return isCurrentPoint;
}

template <typename Tria>
std::unique_ptr<CstSelectorBase<Tria>>
CstFunc<Tria>::clone() const
{
  return std::make_unique<CstFunc<Tria>>(*this);
}


template <typename Tria>
void
CstFunc<Tria>::assignValues(const double         x,
                            const double         y,
                            const double         z,
                            std::vector<double> &values)
{
  if (valuesFunc)
    {
      (*valuesFunc)(x, y, z, values);
    }
  else
    {
      CstSelectorBase<Tria>::assignValues(x, y, z, values);
    }
}



template class bcs::CstFunc<RTria<1, 1>>;
template class bcs::CstFunc<RTria<1, 2>>;
template class bcs::CstFunc<RTria<1, 3>>;


template class bcs::CstFunc<RTria<2, 2>>;
template class bcs::CstFunc<RTria<2, 3>>;

template class bcs::CstFunc<RTria<3, 3>>;


#if defined(DISTRIBUTED_TRIA) && DISTRIBUTED_TRIA
template class bcs::CstFunc<DTria<1, 1>>;
template class bcs::CstFunc<DTria<1, 2>>;
template class bcs::CstFunc<DTria<1, 3>>;


template class bcs::CstFunc<DTria<2, 2>>;
template class bcs::CstFunc<DTria<2, 3>>;

template class bcs::CstFunc<DTria<3, 3>>;
#endif
