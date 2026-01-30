//
//  InhomogeousCstHandler.cpp
//  main
//
//

#include "InhomogeousCstHandler.h"

using namespace dealii;

template <int dim, int spacedim>
InhomogeousCst<dim, spacedim>
::InhomogeousCst(const dealii::Point<spacedim, double>& point,
                 const std::initializer_list<CstValue>& csts,
                 const double tol)
: cstList(csts)
, __point(point)
, __hasSetIndex(false)
, __tol(tol)
{}


template <int dim, int spacedim>
InhomogeousCst<dim, spacedim>
::CstValue::CstValue(const Direction direction,
                     const double    value)
: direction(direction)
, value(value)
{}



 
template <int dim, int spacedim>
bool
InhomogeousCst<dim, spacedim>
::isInRange(const dealii::Point<spacedim, double>& point) const
{
    const bool isInRange = (point.distance(__point) < __tol);
    if(isInRange)
    {
        __hasSetIndex = true;
    }
    return isInRange;
}

template <int dim, int spacedim>
bool
InhomogeousCst<dim, spacedim>
::hasSetIndex() const
{
    return __hasSetIndex;
}


template <int dim, int spacedim>
void
InhomogeousCst<dim, spacedim>
::setIndex(const dealii::DoFHandler<dim, spacedim>& dofHandler,
           const VertIter& vertexIter) const
{
    DoFAccessor<0, dim, spacedim, false>
    vertex_dofs(&(dofHandler.get_triangulation()),
                vertexIter->level(),
                vertexIter->index(),
                &dofHandler);
    
    // extract dof on constrained dof
    for (const CstValue& cstValue : cstList)
    {
        cstValue.idx = vertex_dofs.vertex_dof_index(0, static_cast<unsigned int>(cstValue.direction));
    }
    __hasSetIndex = true;
}

template <int dim, int spacedim>
void
InhomogeousCst<dim, spacedim>
::setIndex(const CellIter& cellIter,
           const unsigned int ithVertex) const
{
    // extract dof on constrained dof
    for (const CstValue& cstValue : cstList)
    {
        cstValue.idx = cellIter->vertex_dof_index(ithVertex,
                                              static_cast<unsigned int>(cstValue.direction));
    }
    __hasSetIndex = true;
}


template class InhomogeousCst<1>;
template class InhomogeousCst<2>;
template class InhomogeousCst<3>;






template < bool isMPI, int dim, int spacedim>
void
InhomogeousCstHandler<isMPI, dim, spacedim>
::setInhomogeousCst(dealii::AffineConstraints<double>&       constraints,
                    const dealii::DoFHandler<dim, spacedim>& dofHandler,
                    const unsigned int nDoFsPerVertex,
                    const InhomoCstList& cstList,
                    const bool atBoundary)
{
    const InhomoCstVector cstVector = cstList;
    
    DoFsOnVertex node_xy(nDoFsPerVertex);
    
    std::vector<DoFsOnVertex> dofsList(cstList.size());
    
    if constexpr (isMPI)
    {
        std::vector<bool> locally_owned_vertices =  GridTools::get_locally_owned_vertices(dofHandler.get_triangulation());
        for (auto const & cell : dofHandler.active_cell_iterators()) {
            if (!cell->is_locally_owned()) continue;
            if (atBoundary && !cell->at_boundary()) continue;
            
            for (const auto vertex : cell->vertex_indices())
            {
                // skip ghost cells
                if (!locally_owned_vertices[cell->vertex_index(vertex)]) continue;
                
                const Point<spacedim> point = cell->vertex(vertex);
                
                for (const InhomoCst& cst : cstVector)
                {
                    // skip the cst that has been set
                    if (cst.hasSetIndex()) continue;
                    // skip the vertices out of given range
                    if (!cst.isInRange(point)) continue;;
                    
                    
                    cst.setIndex(cell, vertex);
                }
            }
        }
        
        for (const InhomoCst& cst : cstVector)
        {
            for (const CstValue& cstValue : cst.cstList)
            {
                if(cstValue.idx != numbers::invalid_dof_index) {
                    constraints.add_line(cstValue.idx);
                    constraints.set_inhomogeneity(cstValue.idx, cstValue.value);
                }
            }
        }
    } else {
        typename Triangulation<dim, spacedim>::active_vertex_iterator vertex_itr;
        vertex_itr = dofHandler.get_triangulation().begin_active_vertex();

        for (; 
             vertex_itr != dofHandler.get_triangulation().end_vertex();
             ++vertex_itr)
        {
            for (const InhomoCst& cst : cstVector)
            {
                // skip the cst that has been set
                if (cst.hasSetIndex()) continue;
                
                const Point<spacedim> point = vertex_itr->vertex();
                
                // skip the vertices out of given range
                if (!cst.isInRange(point)) continue;
                
                
                cst.setIndex(dofHandler, vertex_itr);
            }
            
//            if (   (std::fabs(vertex_itr->vertex()[0] - 25.0) < 1.0e-9)
//                && (std::fabs(vertex_itr->vertex()[1] -  5.0) < 1.0e-9) )
//            {
//                node_xy = usr_utilities::get_vertex_dofs(vertex_itr, dofHandler);
//            }
        }
//        constraints.add_line(node_xy[1]);
//        constraints.set_inhomogeneity(node_xy[1], 0.0);
        for (const InhomoCst& cst : cstVector)
        {
            for (const CstValue& cstValue : cst.cstList)
            {
                if(cstValue.idx != numbers::invalid_dof_index) {
                    constraints.add_line(cstValue.idx);
                    constraints.set_inhomogeneity(cstValue.idx, cstValue.value);
                }
            }
            
        }
    }
}








template class InhomogeousCstHandler<true, 1, 1>;
template class InhomogeousCstHandler<false, 1, 1>;

template class InhomogeousCstHandler<true, 1, 2>;
template class InhomogeousCstHandler<false, 1, 2>;

template class InhomogeousCstHandler<true, 1, 3>;
template class InhomogeousCstHandler<false, 1, 3>;




template class InhomogeousCstHandler<true, 2, 2>;
template class InhomogeousCstHandler<false, 2, 2>;

template class InhomogeousCstHandler<true, 2, 3>;
template class InhomogeousCstHandler<false, 2, 3>;





template class InhomogeousCstHandler<true, 3, 3>;
template class InhomogeousCstHandler<false, 3, 3>;

