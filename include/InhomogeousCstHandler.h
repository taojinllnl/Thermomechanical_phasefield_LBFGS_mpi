//
//  InhomogeousCstHandler.h
//  main
//
//

#ifndef InhomogeousCstHandler_h
#define InhomogeousCstHandler_h

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_tools.h>

#include <vector>

#include <initializer_list>

#include "Traits.h"

#include "Utilities.h"

template <int dim, int spacedim=dim>
class InhomogeousCst {
public:
    enum Direction : short
    {
        x = 0,
        y = 1,
        z = 2,
    };
    
    struct CstValue
    {
        const Direction                 direction;
        const double                    value;
        mutable dealii::types::global_dof_index idx = dealii::numbers::invalid_dof_index;
        
        CstValue(const Direction direction,
                 const double    value);
    };
    
    const std::vector<CstValue>           cstList;
    
private:
    const dealii::Point<spacedim, double> __point;
    

    mutable bool __hasSetIndex;
    const double __tol;
public:
    using VertIter = typename dealii::Triangulation<dim, spacedim>::active_vertex_iterator;
    using CellIter = typename dealii::DoFHandler<dim, spacedim>::active_cell_iterator;
    
    InhomogeousCst(const dealii::Point<spacedim, double>& point,
                   const std::initializer_list<CstValue>& cstList,
                   const double tol = 1.0e-9);
    
    bool isInRange(const dealii::Point<spacedim, double>& point) const;
    bool hasSetIndex() const;
    
    
    void setIndex(const dealii::DoFHandler<dim, spacedim>& dofHandler,
                  const VertIter& vertexIter) const;
    void setIndex(const CellIter& cellIter,
                  const unsigned int ithVertex) const;
};

template <bool isMPI, int dim, int spacedim=dim>
class InhomogeousCstHandler {
    
private:
    using InhomoCst = InhomogeousCst<dim, spacedim>;
    using CstValue  = typename InhomoCst::CstValue;
    using InhomoCstList = std::initializer_list<InhomoCst>;
    using InhomoCstVector = std::vector<InhomoCst>;
    
    using DoFsOnVertex = std::vector<dealii::types::global_dof_index>;
public:
    
    virtual ~InhomogeousCstHandler() = default;
    
    void setInhomogeousCst(dealii::AffineConstraints<double>& constraints,
                           const dealii::DoFHandler<dim, spacedim>& dofHandler,
                           const unsigned int nDoFsPerVertex,
                           const InhomoCstList& cstList,
                           const bool atBoundary = true);
    
   
};



#endif /* InhomogeousCstHandler_h */
