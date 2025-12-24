//
//  OutputHelper.hpp
//  main
//
//

#ifndef OutputHelper_hpp
#define OutputHelper_hpp

#include "Traits.h"
#include "BlockVectorWrapper.h"
#include "MPIInfo.h"
#include "VersionAdapter.h"

#include <memory>
#include <array>

#include <deal.II/base/quadrature_point_data.h>



#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>


namespace PhaseField_monolithic {

template <typename LATraits, typename Tria, typename PointHistory>
class OutputHelper
{
public:
    static constexpr int  dim       = Tria::dimension;
    static constexpr bool is_mpi    =
        !std::is_same_v<typename LATraits::TMTag, ::la::TagSerial>;

    using BVector  = ::la::BlockVectorWrapper<LATraits>;
    
    using CellDataStorage = dealii::CellDataStorage<typename Tria::cell_iterator, PointHistory>;
    
    using DataComponentInterpretationList = std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>;
private:
    const MPIInfo&                      __mpiInfo;
                
    Tria&                               __tria;
    
    const dealii::DoFHandler<dim>&      __dof_handler;
    
    const dealii::QGauss<dim>&          __qf_cell;
    
    const std::vector<std::string>      __solution_name;
    
    const DataComponentInterpretationList __data_component_interpretation;
    
    const DataComponentInterpretationList __data_component_L2;
    
   
    
private:
    static std::vector<std::string> __makeSolutionName();
    
    static DataComponentInterpretationList __makeDataComponentInterpretation();
    
    static DataComponentInterpretationList __makeL2DataComponentInterpretation();
    
    
    void __stressL2(dealii::DataOut<dim>&       data_out,
                    dealii::DoFHandler<dim>&    dof_handler_L2,
                    dealii::AffineConstraints<double>&  constraints,
                    const unsigned int polyDegree,
                    const CellDataStorage& qPntHistory)   const;
    
    void __heatFluxL2(dealii::DataOut<dim>&         data_out,
                      dealii::DoFHandler<dim>&      dof_handler_L2,
                      dealii::DoFHandler<dim>& dof_handler_L2_flux,
                      dealii::AffineConstraints<double>&    constraints,
                      const unsigned int polyDegree,
                      const CellDataStorage& qPntHistory) const;
    
    void __materialIDs(dealii::DataOut<dim>& data_out) const;
    
    void __solution(dealii::DataOut<dim>& data_out,
                    const BVector&     solution) const;
    
    void __partitioning(dealii::DataOut<dim>& data_out) const;
    
public:
    OutputHelper(const MPIInfo&                   mpiInfo,
                 Tria&                            tria,
                 const dealii::DoFHandler<dim>&   dof_handler,
                 const dealii::QGauss<dim>&       qf_cell);
    
    
    void output(const unsigned int ithTimeStep,
                const unsigned int polyDegree,
                const std::string& dir,
                const BVector&     solution,
                const CellDataStorage& qPntHistory) const;
};











template <typename LATraits, typename Tria, typename PointHistory>
std::vector<std::string>
OutputHelper<LATraits, Tria, PointHistory>
::__makeSolutionName()
{
    std::vector<std::string> solution_name(dim, "displacement");
    solution_name.emplace_back("phasefield");
    solution_name.emplace_back("temperature");
    return solution_name;
}

template <typename LATraits, typename Tria, typename PointHistory>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
OutputHelper<LATraits, Tria, PointHistory>
::__makeDataComponentInterpretation()
{
    using namespace dealii;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    return data_component_interpretation;
}


template <typename LATraits, typename Tria, typename PointHistory>
std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation>
OutputHelper<LATraits, Tria, PointHistory>
::__makeL2DataComponentInterpretation()
{
    using namespace dealii;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation_L2(1,
                                     DataComponentInterpretation::component_is_scalar);
    return data_component_interpretation_L2;
}

template <typename LATraits, typename Tria, typename PointHistory>
OutputHelper<LATraits, Tria, PointHistory>
::OutputHelper(const MPIInfo&                   mpiInfo,
               Tria&                            tria,
               const dealii::DoFHandler<dim>&   dof_handler,
               const dealii::QGauss<dim>&       qf_cell)
: __mpiInfo(mpiInfo)
, __tria(tria)
, __dof_handler(dof_handler)
, __qf_cell(qf_cell)
, __solution_name(OutputHelper<LATraits, Tria, PointHistory>::__makeSolutionName())
, __data_component_interpretation(OutputHelper<LATraits, Tria, PointHistory>::__makeDataComponentInterpretation())
, __data_component_L2(OutputHelper<LATraits, Tria, PointHistory>::__makeL2DataComponentInterpretation())
{}



template <typename LATraits, typename Tria, typename PointHistory>
void
OutputHelper<LATraits, Tria, PointHistory>
::__stressL2(dealii::DataOut<dim>& data_out,
             dealii::DoFHandler<dim>& dof_handler_L2,
             dealii::AffineConstraints<double>&  constraints,
             const unsigned int polyDegree,
             const CellDataStorage& qPntHistory) const
{
    using namespace dealii;
    //L2 projection

//    if constexpr (!is_mpi){
//        //stress L2 projection
//        for (unsigned int i = 0; i < dim; ++i)
//            for (unsigned int j = i; j < dim; ++j)
//            {
//                Vector<double> stress_field_L2;
//                stress_field_L2.reinit(dof_handler_L2.n_dofs());
//                
//                MappingQ<dim> mapping(polyDegree + 1);
//                VectorTools::project(mapping,
//                                     dof_handler_L2,
//                                     constraints,
//                                     __qf_cell,
//                                     [&] (const typename DoFHandler<dim>::active_cell_iterator & cell,
//                                          const unsigned int q) -> double
//                                     {
//                    return qPntHistory.get_data(cell)[q]->get_cauchy_stress()[i][j];
//                },
//                                     stress_field_L2);
//                
//                std::string stress_name = "Cauchy_stress_" + std::to_string(i+1) + std::to_string(j+1)
//                + "_L2";
//                
//                data_out.add_data_vector(dof_handler_L2,
//                                         stress_field_L2,
//                                         stress_name,
//                                         __data_component_L2);
//            }
//    } else {
//        //stress L2 projection
//        for (unsigned int i = 0; i < dim; ++i)
//            for (unsigned int j = i; j < dim; ++j)
//            {
//                typename LATraits::VectorBlock stress_field_L2{};
//
//                const auto &locally_owned   = dof_handler_L2.locally_owned_dofs();
//                const auto  locally_relevant =
//                DoFTools::extract_locally_relevant_dofs(dof_handler_L2);
//                
//                stress_field_L2.reinit(locally_owned,
//                                       locally_relevant,
//                                       *__mpiInfo.mpiCommPtr());
//                
//                MappingQ<dim> mapping(polyDegree + 1);
//                VectorTools::project(mapping,
//                                     dof_handler_L2,
//                                     constraints,
//                                     __qf_cell,
//                                     [&] (const typename DoFHandler<dim>::active_cell_iterator & cell,
//                                          const unsigned int q) -> double
//                                     {
//                    return qPntHistory.get_data(cell)[q]->get_cauchy_stress()[i][j];
//                },
//                                     stress_field_L2);
//                
//                std::string stress_name = "Cauchy_stress_" + std::to_string(i+1) + std::to_string(j+1)
//                + "_L2";
//                
//                data_out.add_data_vector(dof_handler_L2,
//                                         stress_field_L2,
//                                         stress_name,
//                                         __data_component_L2);
//            }
//    }

    MappingQ<dim> mapping(polyDegree + 1);
    
    for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = i; j < dim; ++j)
        {
            typename LATraits::VectorBlock stress_field_L2;
            typename LATraits::VectorBlock stress_field_L2_rele;
            
            if constexpr (is_mpi)
            {
                const IndexSet &locally_owned_dofs = dof_handler_L2.locally_owned_dofs();
                const IndexSet locally_relevant_dofs =
                    DoFTools::extract_locally_relevant_dofs(dof_handler_L2);
                
                stress_field_L2.reinit(locally_owned_dofs,
                                       *__mpiInfo.mpiCommPtr());
                stress_field_L2_rele.reinit(locally_owned_dofs,
                                            locally_relevant_dofs,
                                            *__mpiInfo.mpiCommPtr());
            }
            else
            {
                stress_field_L2.reinit(dof_handler_L2.n_dofs());
            }
            
            VectorTools::project(mapping,
                                 dof_handler_L2,
                                 constraints,
                                 __qf_cell,
                                 [&qPntHistory, i, j](const auto &cell, const unsigned int q) -> double {
                if constexpr (is_mpi)
                    if (!cell->is_locally_owned()) 
                        return 0.0;
                return qPntHistory.get_data(cell)[q]->get_cauchy_stress()[i][j];
            },
                                 stress_field_L2);
            
            if constexpr (is_mpi)
            {
                stress_field_L2.compress(dealii::VectorOperation::insert);
                stress_field_L2_rele = stress_field_L2;
                stress_field_L2_rele.update_ghost_values();
            }
            
            const std::string stress_name =
            "Cauchy_stress_" + std::to_string(i + 1) + std::to_string(j + 1) + "_L2";
            
            if constexpr (is_mpi){
                data_out.add_data_vector(dof_handler_L2,
                                         stress_field_L2_rele,
                                         stress_name,
                                         __data_component_L2);
            } else {
                data_out.add_data_vector(dof_handler_L2,
                                         stress_field_L2,
                                         stress_name,
                                         __data_component_L2);
            }
        }
}


template <typename LATraits, typename Tria, typename PointHistory>
void
OutputHelper<LATraits, Tria, PointHistory>
::__heatFluxL2(dealii::DataOut<dim>& data_out,
               dealii::DoFHandler<dim>& dof_handler_L2,
               dealii::DoFHandler<dim>& dof_handler_L2_flux,
               dealii::AffineConstraints<double>&  constraints,
               const unsigned int polyDegree,
               const CellDataStorage& qPntHistory) const
{
    using namespace dealii;
    // Heat flux L2 projection
    
    std::array<typename LATraits::VectorBlock, dim> heat_flux_L2_list;
    std::array<typename LATraits::VectorBlock, dim> heat_flux_L2_rele_list;
//    Vector<double> heat_flux_field_L2_x;
//    Vector<double> heat_flux_field_L2_y;
//    Vector<double> heat_flux_field_L2_z;
    
    for (unsigned int i = 0; i < dim; ++i)
    {
//        typename LATraits::VectorBlock heat_flux_field_L2;
        
        if constexpr (is_mpi){
            const IndexSet &locally_owned_dofs = dof_handler_L2.locally_owned_dofs();
            const IndexSet locally_relevant_dofs =
                DoFTools::extract_locally_relevant_dofs(dof_handler_L2);
            
            heat_flux_L2_list[i].reinit(locally_owned_dofs,
                                        *__mpiInfo.mpiCommPtr());
            heat_flux_L2_rele_list[i].reinit(locally_owned_dofs,
                                             locally_relevant_dofs,
                                             *__mpiInfo.mpiCommPtr());
        } else {
            heat_flux_L2_list[i].reinit(dof_handler_L2.n_dofs());
        }
        
        MappingQ<dim> mapping(polyDegree + 1);
        VectorTools::project(mapping,
                             dof_handler_L2,
                             constraints,
                             __qf_cell,
                             [&] (const typename DoFHandler<dim>::active_cell_iterator & cell,
                                  const unsigned int q) -> double
                             {
            if constexpr (is_mpi)
                if (!cell->is_locally_owned())
                    return 0.0;
            return qPntHistory.get_data(cell)[q]->get_heat_flux()[i];
        },
                             heat_flux_L2_list[i]);
        
        if constexpr (is_mpi) {
            heat_flux_L2_list[i].compress(dealii::VectorOperation::insert);
            heat_flux_L2_rele_list[i] = heat_flux_L2_list[i];
            heat_flux_L2_rele_list[i].update_ghost_values();
        }
        
        std::string heat_flux_name = "Heat_flux_" + std::to_string(i+1) + "_L2";
        if constexpr (is_mpi) {
            data_out.add_data_vector(dof_handler_L2,
                                     heat_flux_L2_rele_list[i],
                                     heat_flux_name,
                                     __data_component_L2);
            
        } else {
        
            data_out.add_data_vector(dof_handler_L2,
                                     heat_flux_L2_list[i],
                                     heat_flux_name,
                                     __data_component_L2);
            
        }
//        heat_flux_L2_list[i] = heat_flux_field_L2;
    }

//    // For 2D problems, let the flux in the third direction is zero
//    if (dim == 2)
//    {
//        typename LATraits::VectorBlock heat_flux_field_L2;
//        
//        if constexpr (is_mpi){
//            heat_flux_field_L2.reinit(*__locally_owned_dofs,
//                                      *__locally_relevant_dofs,
//                                      *__mpiInfo.mpiCommPtr());
//        } else {
//            heat_flux_field_L2.reinit(dof_handler_L2.n_dofs());
//        }
//        
//        heat_flux_field_L2 = 0;
//        std::string heat_flux_name = "Heat_flux_" + std::to_string(3) + "_L2";
//        data_out.add_data_vector(dof_handler_L2,
//                                 heat_flux_field_L2,
//                                 heat_flux_name,
//                                 __data_component_L2);
////        heat_flux_L2_list[2] = 0;
//    }
    
    
    FESystem<dim>   fe_flux_L2(FE_Q<dim>(polyDegree), dim);
    dof_handler_L2_flux.distribute_dofs(fe_flux_L2);
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation_flux_L2(dim,
                                          DataComponentInterpretation::component_is_part_of_vector);
    
    typename LATraits::VectorBlock heat_flux_field_L2;
    typename LATraits::VectorBlock heat_flux_field_L2_rele;
    
    if constexpr (is_mpi){
        
        const IndexSet& owned_dofs = dof_handler_L2_flux.locally_owned_dofs();
        const IndexSet  relevant_dofs =
        DoFTools::extract_locally_relevant_dofs(dof_handler_L2_flux);
        
        
        heat_flux_field_L2.reinit(owned_dofs,
                                  *__mpiInfo.mpiCommPtr());
        heat_flux_field_L2_rele.reinit(owned_dofs,
                                       relevant_dofs,
                                       *__mpiInfo.mpiCommPtr());
        
        const unsigned int dofs_per_cell_scalar = FE_Q<dim>(polyDegree).n_dofs_per_cell();
        const unsigned int dofs_per_cell_vector = fe_flux_L2.n_dofs_per_cell();
        
        std::vector<types::global_dof_index> local_dof_indices_scalar(dofs_per_cell_scalar);
        std::vector<types::global_dof_index> local_dof_indices_vector(dofs_per_cell_vector);
        
        auto cell_scalar = dof_handler_L2.begin_active();
        auto cell_vector = dof_handler_L2_flux.begin_active();
        auto end_it      = dof_handler_L2.end();
        
        for (; cell_scalar != end_it; ++cell_scalar, ++cell_vector) {
            if (cell_scalar->is_locally_owned()) 
            {
            
                cell_scalar->get_dof_indices(local_dof_indices_scalar);
                cell_vector->get_dof_indices(local_dof_indices_vector);
                
                for (unsigned int i = 0; i < dofs_per_cell_scalar; ++i) {
                    for (unsigned int d = 0; d < dim; ++d) {

                        const double value = heat_flux_L2_rele_list[d](local_dof_indices_scalar[i]);
                        heat_flux_field_L2(local_dof_indices_vector[i * dim + d]) =  value;
                    }
                }
            }
        }
        
        heat_flux_field_L2.compress(dealii::VectorOperation::insert);
        
        heat_flux_field_L2_rele = heat_flux_field_L2;
        heat_flux_field_L2_rele.update_ghost_values();
        
        
    } else {
        heat_flux_field_L2.reinit(dof_handler_L2_flux.n_dofs());
        
        for (unsigned int d = 0; d < dim; ++d) {
            for (unsigned int i = 0; i < heat_flux_L2_list[0].size(); ++i)
            {
                heat_flux_field_L2(d+i*dim) = heat_flux_L2_list[d](i);
            }
        }
    }
    
    
    std::vector<std::string> solution_name_flux(dim, "Heat_flux_vector");
    if(is_mpi){
        data_out.add_data_vector(dof_handler_L2_flux,
                                 heat_flux_field_L2_rele,
                                 solution_name_flux,
                                 data_component_interpretation_flux_L2);
    } else {
        data_out.add_data_vector(dof_handler_L2_flux,
                                 heat_flux_field_L2,
                                 solution_name_flux,
                                 data_component_interpretation_flux_L2);
    }
}



template <typename LATraits, typename Tria, typename PointHistory>
void
OutputHelper<LATraits, Tria, PointHistory>
::__materialIDs(dealii::DataOut<dim>& data_out) const
{
    using namespace dealii;
    Vector<float> cell_material_id(__tria.n_active_cells());
    // output material ID for each cell
    
    for (auto cell = __tria.begin_active(); cell != __tria.end(); ++cell)
    {
//        if constexpr (is_mpi)
//        {
//            if(!cell->is_locally_owned())
//            {
//                continue;
//            }
//        }
        cell_material_id(cell->active_cell_index()) = cell->material_id();
    }
    data_out.add_data_vector(cell_material_id, "materialID");
}



template <typename LATraits, typename Tria, typename PointHistory>
void
OutputHelper<LATraits, Tria, PointHistory>
::__solution(dealii::DataOut<dim>& data_out,
             const BVector&         solution) const
{
    using namespace dealii;
    if constexpr (is_mpi) {
        // solution
        data_out.add_data_vector(solution.relevance(),
                                 __solution_name,
                                 DataOut<dim>::type_dof_data,
                                 __data_component_interpretation);
    } else {
        data_out.add_data_vector(solution.base(),
                                 __solution_name,
                                 DataOut<dim>::type_dof_data,
                                 __data_component_interpretation);
    }
}


template <typename LATraits, typename Tria, typename PointHistory>
void
OutputHelper<LATraits, Tria, PointHistory>
::__partitioning(dealii::DataOut<dim>& data_out) const
{
    using namespace dealii;
    if constexpr (is_mpi) {
        // partitioning 1
        Vector<float> subdomain(__tria.n_active_cells());
        for (unsigned int i = 0; i < subdomain.size(); ++i)
            subdomain(i) = __tria.locally_owned_subdomain();
        data_out.add_data_vector(subdomain, "subdomain");
        
        // partitioning 2
        Vector<float> subdomain_cell(__tria.n_active_cells());
        for (auto cell = __tria.begin_active(); cell != __tria.end(); ++cell)
            subdomain_cell(cell->active_cell_index()) = cell->subdomain_id();
        data_out.add_data_vector(subdomain_cell, "partitioning");

    }
}


template <typename LATraits, typename Tria, typename PointHistory>
void OutputHelper<LATraits, Tria, PointHistory>
::output(const unsigned int ithTimeStep,
         const unsigned int polyDegree,
         const std::string& dir,
         const BVector&     solution,
         const CellDataStorage& qPntHistory) const
{
    using namespace dealii;
    
    const std::string filename = "Solution-" + std::to_string(dim) + "d-";
    
    DataOut<dim> data_out;
    data_out.attach_dof_handler(__dof_handler);
    
    
    __materialIDs(data_out);
    
    
    DoFHandler<dim> dof_handler_L2(__tria);
    FE_Q<dim>     fe_L2(polyDegree); //FE_Q element is continuous
    dof_handler_L2.distribute_dofs(fe_L2);
    
    
    AffineConstraints<double> constraints;
    constraints.clear();
    if constexpr (is_mpi)
    {
        const IndexSet &locally_owned_dofs = dof_handler_L2.locally_owned_dofs();
        const IndexSet locally_relevant_dofs =
            DoFTools::extract_locally_relevant_dofs(dof_handler_L2);
        
        constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    }
    DoFTools::make_hanging_node_constraints(dof_handler_L2, constraints);
    constraints.close();
    
    
    
    __solution(data_out, solution);


    __stressL2(data_out,
               dof_handler_L2,
               constraints,
               polyDegree,
               qPntHistory);

    std::cout << "__heatFluxL2; -->" << std::endl;
    DoFHandler<dim> dof_handler_L2_flux(__tria);
    __heatFluxL2(data_out,
                 dof_handler_L2,
                 dof_handler_L2_flux,
                 constraints,
                 polyDegree,
                 qPntHistory);
    std::cout << "__heatFluxL2; -->|" << std::endl;
    std::cout << "__partitioning; -->" << std::endl;
    __partitioning(data_out);
    std::cout << "__partitioning; -->|" << std::endl;
    
    data_out.build_patches(polyDegree);
    
    if constexpr (!is_mpi) {
        std::ofstream output(dir + filename +
                             Utilities::int_to_string(ithTimeStep, 4) + ".vtu");
         
        data_out.write_vtu(output);
    } else {
        data_out.write_vtu_with_pvtu_record(dir,
                                            filename, 
                                            ithTimeStep,
                                            *__mpiInfo.mpiCommPtr(),
                                            4 /*n_digits*/,
                                            0 /*n_groups*/);
    }

}


}
#endif /* OutputHelper_hpp */



