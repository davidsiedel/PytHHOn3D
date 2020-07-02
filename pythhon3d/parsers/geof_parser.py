import numpy as np

C_c2d3 = np.array([[0, 1], [1, 2], [2, 0]])
C_c2d4 = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
C_c3d4 = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 3]])

C_cf_ref = {
    "c2d3": C_c2d3,
    "c2d4": C_c2d4,
    "c3d4": C_c3d4,
}


def parse_geof_file(geof_file_path):
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
    with open(geof_file_path, "r") as geof_file:
        c = geof_file.readlines()
        i_line = 2
        # --------------------------------------------------------------------------------------------------------------
        # reading the dimension of the nodes matrix.
        # --------------------------------------------------------------------------------------------------------------
        N_size_rows = int(c[i_line].rstrip().split(" ")[0])
        N_size_cols = int(c[i_line].rstrip().split(" ")[1])
        problem_dimension = N_size_cols
        # --------------------------------------------------------------------------------------------------------------
        # skipping the line to the first nodes.
        # --------------------------------------------------------------------------------------------------------------
        i_line += 1
        vertices = []
        for i in range(i_line, i_line + N_size_rows):
            vertex = [float(c[i].split(" ")[j]) for j in range(1, N_size_cols + 1)]
            vertices.append(vertex)
        # --------------------------------------------------------------------------------------------------------------
        i_line += N_size_rows + 1
        # --------------------------------------------------------------------------------------------------------------
        # reading the dimension of the C_nc matrix.
        # --------------------------------------------------------------------------------------------------------------
        N_cells = int(c[i_line].rstrip())
        # ==============================================================================================================
        # EXTRACTING THE NSETS MATRICES
        # ==============================================================================================================
        # --------------------------------------------------------------------------------------------------------------
        # skipping the extraction of the cells connectivity matrix to extract
        # directly the Nsets matrices. By doing so, it is easier when defining
        # faces afterward to check whether they belong to a Nset or not. hence,
        # one defines a temporary line increment for the reading of Nsets
        # related lines.
        # --------------------------------------------------------------------------------------------------------------
        i_line_nsets = i_line + N_cells + 2
        Nsets = {}
        for i in range(i_line_nsets, len(c)):
            if "**liset" in c[i] or "**elset" in c[i]:
                break
            else:
                if "**nset" in c[i]:
                    Nset_name = (c[i]).split(" ")[1].rstrip()
                    Nset_nodes = []
                else:
                    if not "***" in c[i]:
                        if c[i].split(" ")[0] == "":
                            start_point = 1
                        else:
                            start_point = 0
                        nodes = [int(item.replace("\n", "")) for item in c[i].split(" ")[start_point:]]
                        Nset_nodes += nodes
                    else:
                        Nsets[Nset_name] = Nset_nodes
                        break
            Nsets[Nset_name] = Nset_nodes
        # ==============================================================================================================
        # EXTRACTING THE CELLS CONNECTIVITY MATRIX
        # ==============================================================================================================
        # --------------------------------------------------------------------------------------------------------------
        # getting back to the line where the connectivity between cells and
        # nodes is defined.
        # --------------------------------------------------------------------------------------------------------------
        i_line += 1
        # --------------------------------------------------------------------------------------------------------------
        # checking weather all elements are of the same nature, i.e. if the mesh
        # contains either c2d3 and c2d4 elements for instance. If so, the cells
        # connectivity matrix does not have a homogeneous number of columns.
        # --------------------------------------------------------------------------------------------------------------
        cells_type_matrix = []
        cells_vertices_connectivity_matrix = []
        for i in range(i_line, i_line + N_cells):
            element_type = str(c[i].split(" ")[1])
            cells_type_matrix.append(element_type)
            cell_vertices = [int(c[i].split(" ")[j]) - 1 for j in range(2, len(c[i].split(" ")))]
            cells_vertices_connectivity_matrix.append(cell_vertices)
        # --------------------------------------------------------------------------------------------------------------
        # Initializing
        # - the faces connectivity matrix C_nf.
        # - the cell-face connectivity matrix C_cf.
        # - a tags list that stores a serial ID for each face, in order not to
        #  add the same face twice in the face connectivity matrix.
        # - a flags vector (with size the number fo faces in the mesh) to store
        # the Nsets each face belongs to.
        # - a weights vector with the weight of each node (the number of cells
        # it belongs to)
        # --------------------------------------------------------------------------------------------------------------
        cell_types = []
        for cell_type in cells_type_matrix:
            if not cell_type in cell_types:
                cell_types.append(cell_type)

        weights = [0 for i in range(len(vertices))]
        tags = []
        flags = []
        cells_faces_connectivity_matrix = []
        faces_vertices_connectivity_matrix = []
        # --------------------------------------------------------------------------------------------------------------
        # For each cell :
        # --------------------------------------------------------------------------------------------------------------
        for i in range(len(cells_vertices_connectivity_matrix)):
            for vertex_index in cells_vertices_connectivity_matrix[i]:
                weights[vertex_index] += 1
            cell_face_connectivity_matrix = []
            cell_type = cells_type_matrix[i]
            # ----------------------------------------------------------------------------------------------------------
            # For each face in the ith cell :
            # ----------------------------------------------------------------------------------------------------------
            for j in range(len(C_cf_ref[cell_type])):
                # ------------------------------------------------------------------------------------------------------
                # Extracting the face connectivity matrix.
                # ------------------------------------------------------------------------------------------------------
                face_vertices_connectivity_matrix = []
                for k in C_cf_ref[cell_type][j]:
                    face_vertices_connectivity_matrix.append(cells_vertices_connectivity_matrix[i][k])
                tag = "".join([str(item) for item in np.sort(face_vertices_connectivity_matrix)])
                # ------------------------------------------------------------------------------------------------------
                # If the face is not stored yet :
                # ------------------------------------------------------------------------------------------------------
                if not tag in tags:
                    # --------------------------------------------------------------------------------------------------
                    # Store it in the face connectivity matrix.
                    # --------------------------------------------------------------------------------------------------
                    flags_local = []
                    faces_vertices_connectivity_matrix.append(face_vertices_connectivity_matrix)
                    tags.append(tag)
                    cell_face_connectivity_matrix.append(len(tags) - 1)
                    # --------------------------------------------------------------------------------------------------
                    # For all Nsets, check whether the face belongs to it or
                    # not. If it belongs to any Nset, append flag with the name
                    # of the Nset, and append "NONE" otherwise.
                    # --------------------------------------------------------------------------------------------------
                    for key in Nsets.keys():
                        count = 0
                        for vertex_index in face_vertices_connectivity_matrix:
                            if vertex_index in Nsets[key]:
                                count += 1
                        if count == len(face_vertices_connectivity_matrix):
                            flags_local.append(key)
                    if not flags_local:
                        flags_local = ["NONE"]
                    flags.append(flags_local)
                # ------------------------------------------------------------------------------------------------------
                # If the face is already stored :
                # ------------------------------------------------------------------------------------------------------
                else:
                    cell_face_connectivity_matrix.append(tags.index(tag))
            cells_faces_connectivity_matrix.append(cell_face_connectivity_matrix)
        print("vertices : {}".format(vertices))
        print("cells_vertices_connectivity_matrix : {}".format(cells_vertices_connectivity_matrix))
        print("faces_vertices_connectivity_matrix : {}".format(faces_vertices_connectivity_matrix))
        print("cells_faces_connectivity_matrix : {}".format(cells_faces_connectivity_matrix))
        print("Nsets : {}".format(Nsets))
        # return N, C_nc, C_nf, C_cf, weights, Nsets, flags
        return (
            problem_dimension,
            vertices,
            cells_vertices_connectivity_matrix,
            faces_vertices_connectivity_matrix,
            cells_faces_connectivity_matrix,
            Nsets,
        )
