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
from core.element import Element
from core.pressure import Pressure
from core.displacement import Displacement
from core.load import Load
from core.condensation import Condensation
from core.integration import Integration
from behaviors.behavior import Behavior
from behaviors.laplacian import Laplacian


def build(
    mesh_file: str, face_polynomial_order: int, cell_polynomial_order: int, operator_type: str,
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
    operators = []
    for i, cell in enumerate(cells):
        local_faces = [faces[j] for j in cells_faces_connectivity_matrix[i]]
        op = get_operator(operator_type, cell, local_faces, cell_basis, face_basis, unknown)
        operators.append(op)
    return (
        vertices,
        faces,
        cells,
        operators,
        cells_faces_connectivity_matrix,
        cells_vertices_connectivity_matrix,
        faces_vertices_connectivity_matrix,
        nsets,
        cell_basis,
        face_basis,
        unknown,
    )


def solve(
    vertices: Mat,
    faces: List[Face],
    cells: List[Cell],
    operators: List[Operator],
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
    number_of_faces = len(faces_vertices_connectivity_matrix)
    number_of_cells = len(cells_vertices_connectivity_matrix)
    total_system_size = number_of_faces * face_basis.basis_dimension * unknown.field_dimension
    global_matrix = np.zeros((total_system_size, total_system_size))
    global_vector = np.zeros((total_system_size,))
    stored_matrices = []
    # ------------------------------------------------------------------------------------------------------------------
    # Global matrix
    # ------------------------------------------------------------------------------------------------------------------
    cells_indices = range(len(cells))
    for cell_index in cells_indices:
        local_cell = cells[cell_index]
        local_faces = [faces[i] for i in cells_faces_connectivity_matrix[cell_index]]
        # --------------------------------------------------------------------------------------------------------------
        # External forces
        # --------------------------------------------------------------------------------------------------------------
        number_of_local_faces = len(local_faces)
        a = unknown.field_dimension * (cell_basis.basis_dimension + number_of_local_faces * face_basis.basis_dimension)
        local_external_forces = np.zeros((a,))
        load_vector = Load(local_cell, cell_basis, unknown, load).load_vector
        l0 = 0
        l1 = unknown.field_dimension * cell_basis.basis_dimension
        local_external_forces[l0:l1] += load_vector
        connectivity = cells_faces_connectivity_matrix[cell_index]
        local_faces_indices = cells_faces_connectivity_matrix[cell_index]
        for local_face_index, global_face_index in enumerate(local_faces_indices):
            face = faces[global_face_index]
            face_reference_frame_transformation_matrix = Operator.get_face_passmat(local_cell, face)
            for boundary_name, nset in zip(nsets, nsets.values()):
                if connectivity[local_face_index] in nset:
                    pressure = boundary_conditions[boundary_name][1]
                    pressure_vector = Pressure(
                        face, face_basis, face_reference_frame_transformation_matrix, unknown, pressure
                    ).pressure_vector
                else:
                    pressure_vector = Pressure(
                        face, face_basis, face_reference_frame_transformation_matrix, unknown
                    ).pressure_vector
            l0 = local_face_index * unknown.field_dimension * face_basis.basis_dimension
            l1 = (local_face_index + 1) * unknown.field_dimension * face_basis.basis_dimension
            local_external_forces[l0:l1] += pressure_vector
        # --------------------------------------------------------------------------------------------------------------
        # Stffness matrix
        # --------------------------------------------------------------------------------------------------------------
        tangent_matrix = tangent_matrices[cell_index]
        tangent_matrix_size = tangent_matrix.shape[0]
        total_size = tangent_matrix_size * cell_basis.basis_dimension
        local_mass_matrix = np.zeros((total_size, total_size))
        for i in range(tangent_matrix.shape[0]):
            for j in range(tangent_matrix.shape[1]):
                behavior = lambda x: 1.0
                m_phi_phi_cell = Integration.get_cell_tangent_matrix_in_cell(local_cell, cell_basis, behavior)
                m = Integration.get_cell_mass_matrix_in_cell(local_cell, cell_basis)
                # ------------------------------------------------------------------------------------------------------
                l0 = i * cell_basis.basis_dimension
                l1 = (i + 1) * cell_basis.basis_dimension
                c0 = j * cell_basis.basis_dimension
                c1 = (j + 1) * cell_basis.basis_dimension
                # ------------------------------------------------------------------------------------------------------
                local_mass_matrix[l0:l1, c0:c1] += m_phi_phi_cell
        local_gradient_operator = operators[cell_index].local_gradient_operator
        local_stiffness_form = local_gradient_operator.T @ local_mass_matrix @ local_gradient_operator
        # --------------------------------------------------------------------------------------------------------------
        # Stabilization matrix
        # --------------------------------------------------------------------------------------------------------------
        local_stabilization_form = operators[cell_index].local_stabilization_form
        # --------------------------------------------------------------------------------------------------------------
        # Local matrix
        # --------------------------------------------------------------------------------------------------------------
        local_matrix = local_stiffness_form + local_stabilization_form
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
        ) = Condensation.get_system_decomposition(local_matrix, local_external_forces, unknown, cell_basis)
        # --------------------------------------------------------------------------------------------------------------
        # Static condensation
        # --------------------------------------------------------------------------------------------------------------
        m_cond, v_cond = Condensation.get_condensated_system(
            m_cell_cell_inv, m_cell_faces, m_faces_cell, m_faces_faces, v_cell, v_faces,
        )
        v_cell, m_cell_faces, m_cell_cell_inv
        stored_matrices.append((v_cell, m_cell_faces, m_cell_cell_inv))
        # --------------------------------------------------------------------------------------------------------------
        # Assembly
        # --------------------------------------------------------------------------------------------------------------
        cell_faces_connectivity_matrix = cells_faces_connectivity_matrix[cell_index]
        for local_index_col, global_index_col in enumerate(cell_faces_connectivity_matrix):
            g0c = global_index_col * face_basis.basis_dimension * unknown.field_dimension
            g1c = (global_index_col + 1) * face_basis.basis_dimension * unknown.field_dimension
            l0c = local_index_col * face_basis.basis_dimension * unknown.field_dimension
            l1c = (local_index_col + 1) * face_basis.basis_dimension * unknown.field_dimension
            global_vector[g0c:g1c] += v_cond[l0c:l1c]
            for local_index_row, global_index_row in enumerate(cell_faces_connectivity_matrix):
                g0r = global_index_row * face_basis.basis_dimension * unknown.field_dimension
                g1r = (global_index_row + 1) * face_basis.basis_dimension * unknown.field_dimension
                l0r = local_index_row * face_basis.basis_dimension * unknown.field_dimension
                l1r = (local_index_row + 1) * face_basis.basis_dimension * unknown.field_dimension
                global_matrix[g0r:g1r, g0c:g1c] += m_cond[l0r:l1r, l0c:l1c]
    # ------------------------------------------------------------------------------------------------------------------
    # Displacement
    # ------------------------------------------------------------------------------------------------------------------
    number_of_constrained_faces = 0
    for boundary_name, nset in zip(nsets, nsets.values()):
        displacement = boundary_conditions[boundary_name][0]
        if not displacement is None:
            for displacement_component in displacement:
                if not displacement_component is None:
                    number_of_constrained_faces += len(nset)
    lagrange_multiplyer_matrix = np.zeros(
        (
            number_of_constrained_faces * face_basis.basis_dimension,
            number_of_faces * unknown.field_dimension * face_basis.basis_dimension,
        )
    )
    h_vector = np.zeros((number_of_constrained_faces * face_basis.basis_dimension,))
    count_2 = 0
    for boundary_name, nset in zip(nsets, nsets.values()):
        for face_global_index in nset:
            face = faces[face_global_index]
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
    # ------------------------------------------------------------------------------------------------------------------
    tech1 = False
    if tech1:
        global_vector_2 = np.zeros((total_system_size + number_of_constrained_faces * face_basis.basis_dimension,))
        global_vector_2[:total_system_size] += global_vector
        global_vector_2[total_system_size:] += h_vector
        global_matrix_2 = np.zeros(
            (
                total_system_size + number_of_constrained_faces * face_basis.basis_dimension,
                total_system_size + number_of_constrained_faces * face_basis.basis_dimension,
            )
        )
        global_matrix_2[:total_system_size, :total_system_size] += global_matrix
        global_matrix_2[:total_system_size, total_system_size:] += lagrange_multiplyer_matrix.T
        global_matrix_2[total_system_size:, :total_system_size] += lagrange_multiplyer_matrix
    else:
        global_vector_2 = np.zeros(
            (total_system_size + 2 * (number_of_constrained_faces * face_basis.basis_dimension),)
        )
        l0 = 0
        l1 = total_system_size
        global_vector_2[l0:l1] += global_vector
        #
        l0 = total_system_size
        l1 = total_system_size + (number_of_constrained_faces * face_basis.basis_dimension)
        global_vector_2[l0:l1] += h_vector
        #
        l0 = total_system_size + (number_of_constrained_faces * face_basis.basis_dimension)
        l1 = total_system_size + 2 * (number_of_constrained_faces * face_basis.basis_dimension)
        global_vector_2[l0:l1] += h_vector
        #
        global_matrix_2 = np.zeros(
            (
                total_system_size + 2 * (number_of_constrained_faces * face_basis.basis_dimension),
                total_system_size + 2 * (number_of_constrained_faces * face_basis.basis_dimension),
            )
        )
        # milieu
        l0 = 0
        l1 = total_system_size
        c0 = 0
        c1 = total_system_size
        global_matrix_2[l0:l1, c0:c1] += global_matrix
        # droite
        l0 = 0
        l1 = total_system_size
        c0 = total_system_size
        c1 = total_system_size + (number_of_constrained_faces * face_basis.basis_dimension)
        global_matrix_2[l0:l1, c0:c1] += lagrange_multiplyer_matrix.T
        # droite
        l0 = 0
        l1 = total_system_size
        c0 = total_system_size + (number_of_constrained_faces * face_basis.basis_dimension)
        c1 = total_system_size + 2 * (number_of_constrained_faces * face_basis.basis_dimension)
        global_matrix_2[l0:l1, c0:c1] += lagrange_multiplyer_matrix.T
        # gauche
        l0 = total_system_size
        l1 = total_system_size + (number_of_constrained_faces * face_basis.basis_dimension)
        c0 = 0
        c1 = total_system_size
        global_matrix_2[l0:l1, c0:c1] += lagrange_multiplyer_matrix
        # gauche
        l0 = total_system_size + (number_of_constrained_faces * face_basis.basis_dimension)
        l1 = total_system_size + 2 * (number_of_constrained_faces * face_basis.basis_dimension)
        c0 = 0
        c1 = total_system_size
        global_matrix_2[l0:l1, c0:c1] += lagrange_multiplyer_matrix
        # ID
        id_lag = np.eye(number_of_constrained_faces * face_basis.basis_dimension)
        l0 = total_system_size
        l1 = total_system_size + (number_of_constrained_faces * face_basis.basis_dimension)
        c0 = total_system_size
        c1 = total_system_size + (number_of_constrained_faces * face_basis.basis_dimension)
        global_matrix_2[l0:l1, c0:c1] += id_lag
        # ##
        l0 = total_system_size
        l1 = total_system_size + (number_of_constrained_faces * face_basis.basis_dimension)
        c0 = total_system_size + (number_of_constrained_faces * face_basis.basis_dimension)
        c1 = total_system_size + 2 * (number_of_constrained_faces * face_basis.basis_dimension)
        global_matrix_2[l0:l1, c0:c1] -= id_lag
        # ##
        l0 = total_system_size + (number_of_constrained_faces * face_basis.basis_dimension)
        l1 = total_system_size + 2 * (number_of_constrained_faces * face_basis.basis_dimension)
        c0 = total_system_size
        c1 = total_system_size + (number_of_constrained_faces * face_basis.basis_dimension)
        global_matrix_2[l0:l1, c0:c1] -= id_lag
        # ##
        l0 = total_system_size + (number_of_constrained_faces * face_basis.basis_dimension)
        l1 = total_system_size + 2 * (number_of_constrained_faces * face_basis.basis_dimension)
        c0 = total_system_size + (number_of_constrained_faces * face_basis.basis_dimension)
        c1 = total_system_size + 2 * (number_of_constrained_faces * face_basis.basis_dimension)
        global_matrix_2[l0:l1, c0:c1] += id_lag

    # ------------------------------------------------------------------------------------------------------------------
    print("global_matrix : \n{}".format(global_matrix_2))
    # ------------------------------------------------------------------------------------------------------------------
    global_solution = np.linalg.solve(global_matrix_2, global_vector_2)
    # global_solution = np.linalg.solve(global_matrix, global_vector)
    # ------------------------------------------------------------------------------------------------------------------
    number_of_vertices = vertices.shape[0]
    vertices_sols = np.zeros((unknown.field_dimension * number_of_vertices,))
    vertices_weights = np.zeros((unknown.field_dimension * number_of_vertices,))
    # ------------------------------------------------------------------------------------------------------------------
    # Global matrix
    # ------------------------------------------------------------------------------------------------------------------
    for cell_index in cells_indices:
        # element = elements[element_index]
        local_cell = cells[cell_index]
        cell_faces_connectivity_matrix = cells_faces_connectivity_matrix[cell_index]
        local_faces = [faces[i] for i in cell_faces_connectivity_matrix]
        a = len(local_faces) * face_basis.basis_dimension * unknown.field_dimension
        x_faces = np.zeros((a,))
        for local_index_col, global_index_col in enumerate(cell_faces_connectivity_matrix):
            g0c = global_index_col * face_basis.basis_dimension * unknown.field_dimension
            g1c = (global_index_col + 1) * face_basis.basis_dimension * unknown.field_dimension
            l0c = local_index_col * face_basis.basis_dimension * unknown.field_dimension
            l1c = (local_index_col + 1) * face_basis.basis_dimension * unknown.field_dimension
            x_faces[l0c:l1c] += global_solution[g0c:g1c]
        # --------------------------------------------------------------------------------------------------------------
        # Global matrix
        # --------------------------------------------------------------------------------------------------------------
        (v_cell, m_cell_faces, m_cell_cell_inv) = stored_matrices[cell_index]
        x_cell = Condensation.get_cell_unknown(m_cell_cell_inv, m_cell_faces, v_cell, x_faces)
        cell_vertices_connectivity_matrix = cells_vertices_connectivity_matrix[cell_index]
        local_vertices = vertices[cell_vertices_connectivity_matrix]
        local_vertices_values = np.zeros(unknown.field_dimension * len(local_vertices),)
        for i, vertex in enumerate(local_vertices):
            phi_vector = np.zeros((unknown.field_dimension * cell_basis.basis_dimension,))
            for field_direction in range(unknown.field_dimension):
                l0 = field_direction * cell_basis.basis_dimension
                l1 = (field_direction + 1) * cell_basis.basis_dimension
                phi_vector[l0:l1] += cell_basis.get_phi_vector(vertex, local_cell.centroid, local_cell.volume)
            vertex_value_vector = phi_vector * x_cell
            vertex_value = np.zeros((unknown.field_dimension,))
            for field_direction in range(unknown.field_dimension):
                l0 = field_direction * cell_basis.basis_dimension
                l1 = (field_direction + 1) * cell_basis.basis_dimension
                vertex_value[field_direction] += np.sum(vertex_value_vector[l0:l1])
            local_vertices_values[i * unknown.field_dimension : (i + 1) * unknown.field_dimension] += vertex_value
            # vertex_value = np.sum(vertex_value_vector)
            # local_vertices_values[i] += vertex_value
            # vertices_sols[cell_vertices_connectivity_matrix] += np.array(local_vertices_values)
            vertices_sols[cell_vertices_connectivity_matrix] += local_vertices_values
            vertices_weights[cell_vertices_connectivity_matrix] += np.ones(local_vertices_values.shape)
    # ------------------------------------------------------------------------------------------------------------------
    # Global matrix
    # ------------------------------------------------------------------------------------------------------------------
    vertices_sols = vertices_sols / vertices_weights
    return vertices, vertices_sols, global_solution


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
