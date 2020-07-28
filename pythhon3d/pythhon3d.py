import argparse
import numpy as np
from typing import List
from typing import Callable
from numpy import ndarray as Mat

from parsers.geof_parser import parse_geof_file as parse_mesh
from parsers.element_types import C_cf_ref
from bases.basis import Basis
from bases.monomial import ScaledMonomial
from core.face import Face
from core.cell import Cell
from core.unknown import Unknown
from core.operators.operator import Operator
from core.operators.hdg import HDG
from core.problem import Problem
from core.element import Element
from core.pressure import Pressure
from core.displacement import Displacement
from core.load import Load
from core.condensation import Condensation
from core.integration import Integration
from behaviors.behavior import Behavior
from behaviors.laplacian import Laplacian


def build(
    mesh_file: str,
    face_polynomial_order: int,
    cell_polynomial_order: int,
    operator_type: str,
    # behavior: Behavior,
    # boundary_conditions: dict,
    # load: List[Callable],
):
    """
    ====================================================================================================================
    Description :
    ====================================================================================================================
    
    ====================================================================================================================
    Parameters :
    ====================================================================================================================
    
    ====================================================================================================================
    Exemple :
    ====================================================================================================================
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Checking polynomial order consistency
    # ------------------------------------------------------------------------------------------------------------------
    is_polynomial_order_consistent(face_polynomial_order, cell_polynomial_order)
    # ------------------------------------------------------------------------------------------------------------------
    # Reading the mesh file, and extracting conectivity matrices
    # ------------------------------------------------------------------------------------------------------------------
    (
        problem_dimension,
        vertices,
        cells_vertices_connectivity_matrix,
        faces_vertices_connectivity_matrix,
        cells_faces_connectivity_matrix,
        cells_connectivity_matrix,
        nsets,
    ) = parse_mesh(mesh_file)
    # ------------------------------------------------------------------------------------------------------------------
    # Writing the vertices matrix as a numpy object
    # ------------------------------------------------------------------------------------------------------------------
    vertices = np.array(vertices)
    # ------------------------------------------------------------------------------------------------------------------
    # Creating unknown object
    # ------------------------------------------------------------------------------------------------------------------
    unknown = Unknown(problem_dimension, problem_dimension, cell_polynomial_order, face_polynomial_order)
    # ------------------------------------------------------------------------------------------------------------------
    # Initilaizing polynomial bases
    # ------------------------------------------------------------------------------------------------------------------
    face_basis = ScaledMonomial(face_polynomial_order, problem_dimension - 1)
    cell_basis = ScaledMonomial(cell_polynomial_order, problem_dimension)
    integration_order = unknown.integration_order
    # ------------------------------------------------------------------------------------------------------------------
    # Initilaizing Face objects
    # ------------------------------------------------------------------------------------------------------------------
    faces = []
    for i, face_vertices_connectivity_matrix in enumerate(faces_vertices_connectivity_matrix):
        face_vertices = vertices[face_vertices_connectivity_matrix]
        face = Face(face_vertices, integration_order)
        # for boundary_name, nset in zip(nsets, nsets.values()):
        #     face_vertices = vertices[face_vertices_connectivity_matrix]
        #     face = Face(face_vertices, integration_order)
        # # ----------------------------------------------------------------------------------------------------------
        # # Checking whether the face belongs to a boundary or not : by scanning the nsets and connecting with the
        # # boundary conditions given as argument of the function
        # # ----------------------------------------------------------------------------------------------------------
        # if i in nset:
        #     displacement = boundary_conditions[boundary_name][0]
        #     pressure = boundary_conditions[boundary_name][1]
        #     face = Face(face_vertices, integration_order, displacement=displacement, pressure=pressure)
        # else:
        #     face = Face(face_vertices, integration_order)
        faces.append(face)
    # ------------------------------------------------------------------------------------------------------------------
    # Initilaizing Cell objects
    # ------------------------------------------------------------------------------------------------------------------
    cells = []
    for cell_vertices_connectivity_matrix, cell_connectivity_matrix in zip(
        cells_vertices_connectivity_matrix, cells_connectivity_matrix
    ):
        cell_vertices = vertices[cell_vertices_connectivity_matrix]
        cell = Cell(cell_vertices, cell_connectivity_matrix, integration_order)
        cells.append(cell)
    # ------------------------------------------------------------------------------------------------------------------
    # Initilaizing Elements objects
    # ------------------------------------------------------------------------------------------------------------------
    elements = []
    for i, cell in enumerate(cells):
        local_faces = [faces[j] for j in cells_faces_connectivity_matrix[i]]
        op = get_operator(operator_type, cell, local_faces, cell_basis, face_basis, unknown)
        local_mass_operator = op.local_mass_operator
        local_stabilization_form = op.local_stabilization_form
        local_gradient_operator = op.local_gradient_operator
        element_vertices = vertices[cells_vertices_connectivity_matrix[i]]
        del op
        # element = Element(
        #     element_vertices, cell, local_faces, local_gradient_operator, local_stabilization_form, local_mass_operator
        # )
        element = Element(cell, local_gradient_operator, local_stabilization_form, local_mass_operator)
        elements.append(element)
    return (
        vertices,
        elements,
        faces,
        cells_faces_connectivity_matrix,
        cells_vertices_connectivity_matrix,
        faces_vertices_connectivity_matrix,
        nsets,
        cell_basis,
        face_basis,
        unknown,
    )


def solve(
    # elements: List[Element],
    # faces: List[face],
    # cells_faces_connectivity_matrix: Mat,
    # cells_vertices_connectivity_matrix: Mat,
    # faces_vertices_connectivity_matrix: Mat,
    # vertices: Mat,
    # nsets: dict,
    # tangent_matrices: List[Mat],
    # boundary_conditions: dict,
    # load: Callable,
    # cell_basis: Basis,
    # face_basis: Basis,
    # number_of_cells: int,
    # number_of_faces: int,
    # unknown: Unknown,
    vertices: Mat,
    elements: List[Element],
    faces: List[Face],
    cells_faces_connectivity_matrix: Mat,
    cells_vertices_connectivity_matrix: Mat,
    faces_vertices_connectivity_matrix: Mat,
    nsets: dict,
    cell_basis: Basis,
    face_basis: Basis,
    unknown: Unknown,
    tangent_matrices: List[Mat],
    boundary_conditions: dict,
    load: Callable,
):
    """
    ====================================================================================================================
    Description :
    ====================================================================================================================
    
    ====================================================================================================================
    Parameters :
    ====================================================================================================================
    
    ====================================================================================================================
    Exemple :
    ====================================================================================================================
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Global matrix
    # ------------------------------------------------------------------------------------------------------------------
    # total_system_size = number_of_faces * face_basis.basis_dimension
    number_of_faces = len(faces_vertices_connectivity_matrix)
    number_of_cells = len(cells_vertices_connectivity_matrix)
    total_system_size = number_of_faces * face_basis.basis_dimension * unknown.field_dimension
    global_matrix = np.zeros((total_system_size, total_system_size))
    global_vector = np.zeros((total_system_size,))
    stored_matrices = []
    # ------------------------------------------------------------------------------------------------------------------
    # Global matrix
    # ------------------------------------------------------------------------------------------------------------------
    elements_indices = range(len(elements))
    for element_index in elements_indices:
        element = elements[element_index]
        # --------------------------------------------------------------------------------------------------------------
        # External forces
        # --------------------------------------------------------------------------------------------------------------
        local_faces = [faces[i] for i in cells_faces_connectivity_matrix[element_index]]
        number_of_element_faces = len(local_faces)
        a = unknown.field_dimension * (
            cell_basis.basis_dimension + number_of_element_faces * face_basis.basis_dimension
        )
        local_external_forces = np.zeros((a,))
        element = elements[element_index]
        load_vector = Load(element.cell, cell_basis, unknown, load).load_vector
        l0 = 0
        l1 = unknown.field_dimension * cell_basis.basis_dimension
        # print("load_vector : {}".format(load_vector))
        # print("local_external_forces : {}".format(local_external_forces))
        # print(local_external_forces)
        local_external_forces[l0:l1] += load_vector
        connectivity = cells_faces_connectivity_matrix[element_index]
        local_faces_indices = cells_faces_connectivity_matrix[element_index]
        # for face_index, face in enumerate(element.faces):
        for face_index, global_face_index in enumerate(local_faces_indices):
            face = faces[global_face_index]
            face_reference_frame_transformation_matrix = Operator.get_face_passmat(element.cell, face)
            for boundary_name, nset in zip(nsets, nsets.values()):
                if connectivity[face_index] in nset:
                    pressure = boundary_conditions[boundary_name][1]
                    pressure_vector = Pressure(
                        face, face_basis, face_reference_frame_transformation_matrix, unknown, pressure
                    ).pressure_vector
                else:
                    pressure_vector = Pressure(
                        face, face_basis, face_reference_frame_transformation_matrix, unknown
                    ).pressure_vector
            l0 = face_index * unknown.field_dimension * face_basis.basis_dimension
            l1 = (face_index + 1) * unknown.field_dimension * face_basis.basis_dimension
            local_external_forces[l0:l1] += pressure_vector
        # --------------------------------------------------------------------------------------------------------------
        # Stffness matrix
        # --------------------------------------------------------------------------------------------------------------
        tangent_matrix = tangent_matrices[element_index]
        dim_c = cell_basis.basis_dimension
        tangent_matrix_size = tangent_matrix.shape[0]
        total_size = tangent_matrix_size * dim_c
        local_mass_matrix = np.zeros((total_size, total_size))
        for i in range(tangent_matrix.shape[0]):
            for j in range(tangent_matrix.shape[1]):
                behavior = lambda x: 1.0
                m_phi_phi_cell = Integration.get_cell_tangent_matrix_in_cell(element.cell, cell_basis, behavior)
                m = Integration.get_cell_mass_matrix_in_cell(element.cell, cell_basis)
                # ------------------------------------------------------------------------------------------------------
                l0 = i * dim_c
                l1 = (i + 1) * dim_c
                c0 = j * dim_c
                c1 = (j + 1) * dim_c
                # ------------------------------------------------------------------------------------------------------
                # local_mass_matrix[l0:l1, c0:c1] += tangent_matrix[i, j] * m_phi_phi_cell
                local_mass_matrix[l0:l1, c0:c1] += m_phi_phi_cell
        local_gradient_operator = element.local_gradient_operator
        # print("local_gradient_operator : {}".format(local_gradient_operator))
        # print("local_stab_operator : {}".format(element.local_stabilization_form))
        local_stiffness_form = local_gradient_operator.T @ local_mass_matrix @ local_gradient_operator
        # --------------------------------------------------------------------------------------------------------------
        # Stabilization matrix
        # --------------------------------------------------------------------------------------------------------------
        local_stabilization_form = element.local_stabilization_form
        # --------------------------------------------------------------------------------------------------------------
        # Element matrix
        # --------------------------------------------------------------------------------------------------------------
        local_element_matrix = local_stiffness_form + local_stabilization_form
        # --------------------------------------------------------------------------------------------------------------
        # Static condensation
        # --------------------------------------------------------------------------------------------------------------
        (
            m_cell_cell_inv,
            m_cell_faces,
            m_faces_cell,
            m_faces_faces,
            v_cell,
            v_faces,
        ) = Condensation.get_system_decomposition(local_element_matrix, local_external_forces, unknown, cell_basis)
        # --------------------------------------------------------------------------------------------------------------
        # Static condensation
        # --------------------------------------------------------------------------------------------------------------
        # print("local_external_forces : \n{} {}".format(len(local_external_forces), local_external_forces))
        m_cond, v_cond = Condensation.get_condensated_system(
            m_cell_cell_inv, m_cell_faces, m_faces_cell, m_faces_faces, v_cell, v_faces,
        )
        v_cell, m_cell_faces, m_cell_cell_inv
        stored_matrices.append((v_cell, m_cell_faces, m_cell_cell_inv))
        # # --------------------------------------------------------------------------------------------------------------
        # # Static condensation
        # # --------------------------------------------------------------------------------------------------------------
        # x_faces = np.linalg.solve(m_cond, v_cond)
        # # --------------------------------------------------------------------------------------------------------------
        # # Static condensation
        # # --------------------------------------------------------------------------------------------------------------
        # x_cell = Condensation.get_cell_unknown(m_cell_cell_inv, m_cell_faces, v_cell, x_faces)
        cell_faces_connectivity_matrix = cells_faces_connectivity_matrix[element_index]
        for local_index_col, global_index_col in enumerate(cell_faces_connectivity_matrix):
            # g0c = global_index_col * face_basis.basis_dimension
            # g1c = (global_index_col + 1) * face_basis.basis_dimension
            # l0c = local_index_col * face_basis.basis_dimension
            # l1c = (local_index_col + 1) * face_basis.basis_dimension
            #
            g0c = global_index_col * face_basis.basis_dimension * unknown.field_dimension
            g1c = (global_index_col + 1) * face_basis.basis_dimension * unknown.field_dimension
            l0c = local_index_col * face_basis.basis_dimension * unknown.field_dimension
            l1c = (local_index_col + 1) * face_basis.basis_dimension * unknown.field_dimension
            global_vector[g0c:g1c] += v_cond[l0c:l1c]
            for local_index_row, global_index_row in enumerate(cell_faces_connectivity_matrix):
                # g0r = global_index_row * face_basis.basis_dimension
                # g1r = (global_index_row + 1) * face_basis.basis_dimension
                # l0r = local_index_row * face_basis.basis_dimension
                # l1r = (local_index_row + 1) * face_basis.basis_dimension
                #
                g0r = global_index_row * face_basis.basis_dimension * unknown.field_dimension
                g1r = (global_index_row + 1) * face_basis.basis_dimension * unknown.field_dimension
                l0r = local_index_row * face_basis.basis_dimension * unknown.field_dimension
                l1r = (local_index_row + 1) * face_basis.basis_dimension * unknown.field_dimension
                global_matrix[g0r:g1r, g0c:g1c] += m_cond[l0r:l1r, l0c:l1c]
    # ------------------------------------------------------------------------------------------------------------------
    # Displacement
    # ------------------------------------------------------------------------------------------------------------------
    count = 0
    for boundary_name, nset in zip(nsets, nsets.values()):
        displacement = boundary_conditions[boundary_name][0]
        for displacement_component in displacement:
            if not displacement_component is None:
                count += len(nset)
    lagrange_multiplyer_matrix = np.zeros(
        (count * face_basis.basis_dimension, number_of_faces * unknown.field_dimension * face_basis.basis_dimension)
    )
    h_vector = np.zeros((count * face_basis.basis_dimension,))
    count_2 = 0
    for boundary_name, nset in zip(nsets, nsets.values()):
        for face_global_index in nset:
            face = faces[face_global_index]
            # face_reference_frame_transformation_matrix = Operator.get_face_passmat(element.cell, face)
            face_reference_frame_transformation_matrix = face.reference_frame_transformation_matrix
            displacement = boundary_conditions[boundary_name][0]
            for direction, displacement_component in enumerate(displacement):
                if not displacement_component is None:
                    displacement_vector = Displacement(
                        face, face_basis, face_reference_frame_transformation_matrix, displacement_component
                    ).displacement_vector
                    # --------------------------------------------------------------------------------------------------
                    l0 = count_2 * face_basis.basis_dimension
                    l1 = (count_2 + 1) * face_basis.basis_dimension
                    c0 = (
                        face_global_index * unknown.field_dimension * face_basis.basis_dimension
                        + direction * face_basis.basis_dimension
                    )
                    c1 = (face_global_index * unknown.field_dimension * face_basis.basis_dimension) + (
                        direction + 1
                    ) * face_basis.basis_dimension
                    # --------------------------------------------------------------------------------------------------
                    lagrange_multiplyer_matrix[l0:l1, c0:c1] += np.eye(face_basis.basis_dimension)
                    # --------------------------------------------------------------------------------------------------
                    h_vector[l0:l1] += displacement_vector
                    count_2 += 1
    # global_vector_2 = np.concatenate((global_vector, h_vector))
    global_vector_2 = np.zeros((total_system_size + count * face_basis.basis_dimension,))
    global_vector_2[:total_system_size] += global_vector
    global_vector_2[total_system_size:] += h_vector
    global_matrix_2 = np.zeros(
        (total_system_size + count * face_basis.basis_dimension, total_system_size + count * face_basis.basis_dimension)
    )
    global_matrix_2[:total_system_size, :total_system_size] += global_matrix
    global_matrix_2[:total_system_size, total_system_size:] += lagrange_multiplyer_matrix.T
    global_matrix_2[total_system_size:, :total_system_size] += lagrange_multiplyer_matrix
    # global_solution = np.linalg.solve(global_matrix, global_vector)
    global_solution = np.linalg.solve(global_matrix_2, global_vector_2)
    # print("global_solution : \n{}".format(global_solution))
    print("global_matrix : \n{}".format(global_matrix_2))
    number_of_vertices = vertices.shape[0]
    vertices_sols = np.zeros((number_of_vertices,))
    vertices_weights = np.zeros((number_of_vertices,))
    # ------------------------------------------------------------------------------------------------------------------
    # Global matrix
    # ------------------------------------------------------------------------------------------------------------------
    for element_index in elements_indices:
        element = elements[element_index]
        local_faces = [faces[i] for i in cells_faces_connectivity_matrix[element_index]]
        a = len(local_faces) * face_basis.basis_dimension * unknown.field_dimension
        x_faces = np.zeros((a,))
        cell_faces_connectivity_matrix = cells_faces_connectivity_matrix[element_index]
        for local_index_col, global_index_col in enumerate(cell_faces_connectivity_matrix):
            # g0c = global_index_col * face_basis.basis_dimension
            # g1c = (global_index_col + 1) * face_basis.basis_dimension
            # l0c = local_index_col * face_basis.basis_dimension
            # l1c = (local_index_col + 1) * face_basis.basis_dimension
            #
            g0c = global_index_col * face_basis.basis_dimension * unknown.field_dimension
            g1c = (global_index_col + 1) * face_basis.basis_dimension * unknown.field_dimension
            l0c = local_index_col * face_basis.basis_dimension * unknown.field_dimension
            l1c = (local_index_col + 1) * face_basis.basis_dimension * unknown.field_dimension
            # print("x_faces : \n{}".format(x_faces))
            # print(
            #     "indices : \n{}, {}, {}, {}, len(global_solution) : {}".format(g0c, g1c, l0c, l1c, len(global_solution))
            # )
            # print("global_solution shape : \n{}".format(global_solution.shape))
            x_faces[l0c:l1c] += global_solution[g0c:g1c]
        # --------------------------------------------------------------------------------------------------------------
        # Global matrix
        # --------------------------------------------------------------------------------------------------------------
        (v_cell, m_cell_faces, m_cell_cell_inv) = stored_matrices[element_index]
        x_cell = Condensation.get_cell_unknown(m_cell_cell_inv, m_cell_faces, v_cell, x_faces)
        cell_vertices_connectivity_matrix = cells_vertices_connectivity_matrix[element_index]
        local_vertices = vertices[cells_vertices_connectivity_matrix[element_index]]
        local_vertices_values = np.zeros(len(local_vertices),)
        for i, vertex in enumerate(local_vertices):
            # print("vertex : \n{}".format(vertex))
            # print("centroid : \n{}".format(element.cell.centroid))
            # print("volume : \n{}".format(element.cell.volume))
            phi_vector = np.zeros((unknown.field_dimension * cell_basis.basis_dimension,))
            for field_direction in range(unknown.field_dimension):
                l0 = field_direction * cell_basis.basis_dimension
                l1 = (field_direction + 1) * cell_basis.basis_dimension
                phi_vector[l0:l1] += cell_basis.get_phi_vector(vertex, element.cell.centroid, element.cell.volume)
            # phsi_vector = cell_basis.get_phi_vector(vertex, element.cell.centroid, element.cell.volume)
            vertex_value_vector = phi_vector * x_cell
            vertex_value = np.sum(vertex_value_vector)
            # local_vertices_values[i] += np.sum(vertex_value_vector)
            # vertex_value = np.prod(vertex_value_vector, axis=1,)
            local_vertices_values[i] += vertex_value
            # print("vertices : \n{}".format(vertices))
            # print("vertices_sols : \n{}".format(vertices_sols))
            # print("cell_vertices_connectivity_matrix : \n{}".format(cell_vertices_connectivity_matrix))
            # print("global_solution : {}".format(global_solution))
            vertices_sols[cell_vertices_connectivity_matrix] += np.array(local_vertices_values)
            vertices_weights[cell_vertices_connectivity_matrix] += np.ones(local_vertices_values.shape)
    # ------------------------------------------------------------------------------------------------------------------
    # Global matrix
    # ------------------------------------------------------------------------------------------------------------------
    vertices_sols = vertices_sols / vertices_weights
    import matplotlib.pyplot as plt

    plt.plot(vertices, vertices_sols)
    plt.show()
    return vertices, vertices_sols


# dim_c = cell_basis.basis_dimension
# dim_f = face_basis.basis_dimension
# dim_u = unknown.field_dimension
# number_of_cells = len(cells)
# number_of_faces = len(faces)
# total_size = dim_u * (dim_c * number_of_cells + dim_f * number_of_faces)
# global_matrix = np.zeros((total_size, total_size))
# global_vector = np.zeros((total_size,))
# for element in elements:

#     #
#     pb = Problem(laplacian, op)
#     problems.append(pb)
#     # --------------------------------------------------------------------------------------------------------------
#     # local_cell_mass_matrix = op.local_cell_mass_matrix
#     # # local_face_mass_matrix = op.local_face_mass_matrix
#     # local_identity_operator = op.local_identity_operator
#     # local_reconstructed_gradient_operators = op.local_reconstructed_gradient_operators
#     # local_stabilization_matrix = op.local_stabilization_matrix
#     # local_load_vectors = op.local_load_vectors
#     # local_pressure_vectors = op.local_pressure_vectors
#     # --------------------------------------------------------------------------------------------------------------
#     del op
#     vertices = cells_vertices_connectivity_matrix[i]
#     quadrature_points: cell.quadrature_nodes
#     # del cell
#     # --------------------------------------------------------------------------------------------------------------
#     matrix = pb.local_element_matrix
#     vector = pb.local_element_vector
#     # --------------------------------------------------------------------------------------------------------------
#     lg0 = i * (dim_c * dim_u)
#     lg1 = (i + 1) * (dim_c * dim_u)
#     # --------------------------------------------------------------------------------------------------------------
#     global_matrix[lg0:lg1, lg0:lg1] += matrix[: (dim_c * dim_u), : (dim_c * dim_u)]
#     global_vector[lg0:lg1] += vector[: (dim_c * dim_u)]
#     # --------------------------------------------------------------------------------------------------------------
#     for local_face_j, j in enumerate(cells_faces_connectivity_matrix[i]):
#         lvl0 = (local_face_j) * (dim_f * dim_u)
#         lvl1 = (local_face_j + 1) * (dim_f * dim_u)
#         lvg0 = (j) * (dim_f * dim_u)
#         lvg1 = (j + 1) * (dim_f * dim_u)
#         # ----------------------------------------------------------------------------------------------------------
#         global_vector[lvg0:lvg1] += vector[lvl0:lvl1]
#         # ----------------------------------------------------------------------------------------------------------
#         for local_face_k, k in enumerate(cells_faces_connectivity_matrix[i]):
#             lmg0 = number_of_cells * (dim_c * dim_u) + j * (dim_f * dim_u)
#             lmg1 = number_of_cells * (dim_c * dim_u) + (j + 1) * (dim_f * dim_u)
#             cmg0 = number_of_cells * (dim_c * dim_u) + k * (dim_f * dim_u)
#             cmg1 = number_of_cells * (dim_c * dim_u) + (k + 1) * (dim_f * dim_u)
#             lml0 = (dim_c * dim_u) + local_face_j * (dim_f * dim_u)
#             lml1 = (dim_c * dim_u) + (local_face_j + 1) * (dim_f * dim_u)
#             cml0 = (dim_c * dim_u) + local_face_k * (dim_f * dim_u)
#             cml1 = (dim_c * dim_u) + (local_face_k + 1) * (dim_f * dim_u)
#             # ------------------------------------------------------------------------------------------------------
#             global_matrix[lmg0:lmg1, cmg0:cmg1] += matrix[lml0:lml1, cml0:cml1]
#             # ------------------------------------------------------------------------------------------------------
# print(global_matrix)
# print(global_vector)


def is_polynomial_order_consistent(face_polynomial_order: int, cell_polynomial_order: int):
    """
    ====================================================================================================================
    Description :
    ====================================================================================================================
    
    ====================================================================================================================
    Parameters :
    ====================================================================================================================
    
    ====================================================================================================================
    Exemple :
    ====================================================================================================================
    """
    if not face_polynomial_order in [cell_polynomial_order - 1, cell_polynomial_order, cell_polynomial_order + 1]:
        raise ValueError(
            "The face polynomial order must be the same order as the cell polynomial order or one order lower or greater"
        )


def get_operator(
    operator_type: str, cell: Cell, faces: List[Face], cell_basis: Basis, face_basis: Basis, unknown: Unknown,
):
    """
        ================================================================================================================
        Description :
        ================================================================================================================
        
        ================================================================================================================
        Parameters :
        ================================================================================================================
        
        ================================================================================================================
        Exemple :
        ================================================================================================================
        """
    if operator_type == "HDG":
        op = HDG(cell, faces, cell_basis, face_basis, unknown)
    else:
        raise NameError("The specified operator does not exist")
    return op


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Solve a mechanical system using the HHO method")
#     parser.add_argument("-msh", "--mesh_file", help="shows output")
#     parser.add_argument("-kf", "--face_polynomial_order", help="shows output")
#     parser.add_argument("-kc", "--cell_polynomial_order", help="shows output")
#     parser.add_argument("-op", "--operator_type", help="shows output")
#     args = parser.parse_args()
#     # ------------------------------------------------------------------------------------------------------------------
#     mesh_file = args.mesh_file
#     face_polynomial_order = int(args.face_polynomial_order)
#     cell_polynomial_order = int(args.cell_polynomial_order)
#     operator_type = args.operator_type
#     # ------------------------------------------------------------------------------------------------------------------
#     main(mesh_file, face_polynomial_order, cell_polynomial_order, operator_type)


# #     if
# # # for i in range(number_of_faces):
# # #     for boundary_name, nset in zip(nsets, nsets.values()):
# # for i, face_vertices_connectivity_matrix in enumerate(faces_vertices_connectivity_matrix):
# #     for boundary_name, nset in zip(nsets, nsets.values()):
# #         if i in  nset:
# #             displacement = boundary_conditions[boundary_name][0]

# #             #
# #             for face_index, face in enumerate(element.faces):
# #         face_reference_frame_transformation_matrix = Operator.get_face_passmat(element.cell, face)
# #         for boundary_name, nset in zip(nsets, nsets.values()):
# #             if connectivity[face_index] in nset:
# #                 pressure = boundary_conditions[boundary_name][1]
# #                 pressure_vector = Pressure(
# #                     face, face_basis, face_reference_frame_transformation_matrix, unknown, pressure
# #                 ).pressure_vector
# #             else:
# #                 pressure_vector = Pressure(
# #                     face, face_basis, face_reference_frame_transformation_matrix, unknown
# #                 ).pressure_vector
# #             #

# face_vertices = vertices[face_vertices_connectivity_matrix]
# face = Face(face_vertices, integration_order)
# # for boundary_name, nset in zip(nsets, nsets.values()):
# #     face_vertices = vertices[face_vertices_connectivity_matrix]
# #     face = Face(face_vertices, integration_order)
# # # ----------------------------------------------------------------------------------------------------------
# # # Checking whether the face belongs to a boundary or not : by scanning the nsets and connecting with the
# # # boundary conditions given as argument of the function
# # # ----------------------------------------------------------------------------------------------------------
# # if i in nset:
# #     displacement = boundary_conditions[boundary_name][0]
# #     pressure = boundary_conditions[boundary_name][1]
# #     face = Face(face_vertices, integration_order, displacement=displacement, pressure=pressure)
# # else:
# #     face = Face(face_vertices, integration_order)
# faces.append(face)

