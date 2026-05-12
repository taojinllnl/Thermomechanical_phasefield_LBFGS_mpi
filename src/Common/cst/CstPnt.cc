//
//  CstPnt.cpp
//  main
//
//

#include "../../../include/Common/cst/CstPnt.h"


using namespace ::dealii;
using namespace ::bcs;


template <typename Tria>
CstPnt<Tria>::CstPnt(const std::array<double, 3>       &pntCoorinates,
                     const std::vector<CstEntry<Tria>> &csts,
                     const double                       tol)
  : CstSelectorBase<Tria>(csts)
  , pntCoorinates(std::array<double, 3>(pntCoorinates))
  , valuesFunc(nullptr)
  , tol(tol)
{}


template <typename Tria>
CstPnt<Tria>::CstPnt(const CstPnt &cstPnt)
  : CstSelectorBase<Tria>(cstPnt)
  , pntCoorinates(cstPnt.pntCoorinates)
  , valuesFunc(nullptr)
  , tol(cstPnt.tol)
{}


template <typename Tria>
CstPnt<Tria>::CstPnt(const std::array<double, 3> &pntCoorinates,
                     const CstEntry<Tria>        &csts,
                     const double                 tol)
  : CstSelectorBase<Tria>({{csts}})
  , pntCoorinates(std::array<double, 3>(pntCoorinates))
  , valuesFunc(nullptr)
  , tol(tol)
{}


template <typename Tria>
CstPnt<Tria>::CstPnt(const std::array<double, 3>       &pntCoorinates,
                     const std::vector<CstEntry<Tria>> &csts,
                     const ValuesAtPntFunc             &valuesFunc,
                     const double                       tol)
  : CstSelectorBase<Tria>(csts)
  , pntCoorinates(std::array<double, 3>(pntCoorinates))
  , valuesFunc(std::make_shared<ValuesAtPntFunc>(valuesFunc))
  , tol(tol)
{}


template <typename Tria>
CstPnt<Tria>::CstPnt(const std::array<double, 3> &pntCoorinates,
                     const CstEntry<Tria>        &csts,
                     const ValuesAtPntFunc       &valuesFunc,
                     const double                 tol)
  : CstSelectorBase<Tria>({{csts}})
  , pntCoorinates(std::array<double, 3>(pntCoorinates))
  , valuesFunc(std::make_shared<ValuesAtPntFunc>(valuesFunc))
  , tol(tol)
{}

template <typename Tria>
std::size_t
CstPnt<Tria>::expectedNumberOfCstPoints() const
{
  return 1;
}


template <typename Tria>
std::size_t
CstPnt<Tria>::expectedNumberOfCstEntries() const
{
  return CstSelectorBase<Tria>::cstInput().size();
}


template <typename Tria>
void
CstPnt<Tria>::assignValues(const double         x,
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

template <typename Tria>
bool
CstPnt<Tria>::isSelectedPnt(const ::dealii::Point<spacedim> &point)
{
  bool isCurrentPoint = false;

  if constexpr (spacedim == 1)
    {
      isCurrentPoint = std::fabs(point[0] - pntCoorinates[0]) < tol;
    }
  else if constexpr (spacedim == 2)
    {
      isCurrentPoint = std::fabs(point[0] - pntCoorinates[0]) < tol &&
                       std::fabs(point[1] - pntCoorinates[1]) < tol;
    }
  else if constexpr (spacedim == 3)
    {
      isCurrentPoint = std::fabs(point[0] - pntCoorinates[0]) < tol &&
                       std::fabs(point[1] - pntCoorinates[1]) < tol &&
                       std::fabs(point[2] - pntCoorinates[2]) < tol;
    }
  return isCurrentPoint;
}


template <typename Tria>
std::unique_ptr<CstSelectorBase<Tria>>
CstPnt<Tria>::clone() const
{
  return std::make_unique<CstPnt<Tria>>(*this);
}


template class bcs::CstPnt<RTria<1, 1>>;
template class bcs::CstPnt<RTria<1, 2>>;
template class bcs::CstPnt<RTria<1, 3>>;

template class bcs::CstPnt<RTria<2, 2>>;
template class bcs::CstPnt<RTria<2, 3>>;

template class bcs::CstPnt<RTria<3, 3>>;



#if defined(DISTRIBUTED_TRIA) && DISTRIBUTED_TRIA
template class bcs::CstPnt<DTria<1, 1>>;
template class bcs::CstPnt<DTria<1, 2>>;
template class bcs::CstPnt<DTria<1, 3>>;

template class bcs::CstPnt<DTria<2, 2>>;
template class bcs::CstPnt<DTria<2, 3>>;

template class bcs::CstPnt<DTria<3, 3>>;
#endif
