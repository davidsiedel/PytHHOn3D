import numpy as np

C_c2d3 = np.array([[0, 1], [1, 2], [2, 0]])
C_c2d4 = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

C_c3d4 = np.array([[0, 2, 1], [0, 3, 2], [0, 1, 3]])

C_cf_ref = {
    "c2d3": C_c2d3,
    "c2d4": C_c2d4,
    "c3d4": C_c3d4,
    "c3d6": C_c3d6,
    "c3d8": C_c3d8,
}


def parse_geof_file(geof_file_path):
    with open(geof_file_path, "r") as geof_file:
        c = geof_file.readlines()
        i_line = 2
        # ----------------------------------------------------------------------
        # reading the dimension of the nodes matrix.
        # ----------------------------------------------------------------------
        N_size_rows = int(c[i_line].rstrip().split(" ")[0])
        N_size_cols = int(c[i_line].rstrip().split(" ")[1])
        # ----------------------------------------------------------------------
        # skipping the line to the first nodes.
        # ----------------------------------------------------------------------
        i_line += 1
        vertices = []
        for i in range(i_line, N_size_rows):
            vertex = [float(c[i_line].split(" ")[j]) for j in range(N_size_cols)]
            vertices.append(vertex)
        # --------------------------------------------------------------------------------------------------------------
        i_line += N_size_rows + 1
        # --------------------------------------------------------------------------------------------------------------
        # reading the dimension of the C_nc matrix.
        # ----------------------------------------------------------------------
        N_cells = int(c[i_line].rstrip())
        # ######################################################################
        # EXTRACTING THE NSETS MATRICES
        # ######################################################################
        # ----------------------------------------------------------------------
        # skipping the extraction of the cells connectivity matrix to extract
        # directly the Nsets matrices. By doing so, it is easier when defining
        # faces afterward to check whether they belong to a Nset or not. hence,
        # one defines a temporary line increment for the reading of Nsets
        # related lines.
        # ----------------------------------------------------------------------
        i_line_nsets = i_line + N_cells + 2
        Nsets = {}
        for i in range(i_line_nsets, len(c)):
            if "**nset" in c[i]:
                Nset_name = (c[i]).split(" ")[1].rstrip()
                Nset_nodes = []
            else:
                if not "***" in c[i]:
                    nodes = [int(item) for item in c[i].split(" ")]
                    Nset_nodes += nodes
                else:
                    Nsets[Nset_name] = np.array(Nset_nodes)
                    break
            Nsets[Nset_name] = np.array(Nset_nodes)
        # ######################################################################
        # EXTRACTING THE CELLS CONNECTIVITY MATRIX
        # ######################################################################
        # ----------------------------------------------------------------------
        # getting back to the line where the connectivity between cells and
        # nodes is defined.
        # ----------------------------------------------------------------------
        i_line += 1
        # ----------------------------------------------------------------------
        # checking weather all elements are of the same nature, i.e. if the mesh
        # contains either c2d3 and c2d4 elements for instance. If so, the cells
        # connectivity matrix does not have a homogeneous number of columns.
        # ----------------------------------------------------------------------
        cells_type_matrix = []
        cells_vertices_adjacency_matrix = []
        for i in range(i_line, i_line + N_cells):
            element_type = str(c[i_line].split(" ")[1])
            cells_type_matrix.append(element_type)
            vertices = [int(c[i_line].split(" ")[j])-1 for j in range(2, len(c[i_line].split(" ")))]
            cells_vertices_adjacency_matrix.append(vertices)



        # element_types = np.genfromtxt(c[i_line : i_line + N_cells], usecols=(1), converters={1: lambda s: str(s)})

        # # ----------------------------------------------------------------------
        # # if not all elements are of the same nature :
        # # ----------------------------------------------------------------------
        # element_types_unique = np.unique(element_types)
        # element_type = lambda index: element_types_unique[index][2:-1]
        # if not len(element_types_unique) == 1:
        #     # ------------------------------------------------------------------
        #     # The number of columns (of nodes belonging to the cell) for the
        #     # local cell connectivity matrix C_nc_loc is given by the element
        #     # type (in particular the last char). For instance, d2d4 has 4
        #     # columns.
        #     # ------------------------------------------------------------------
        #     n_cols_temp = [int(element_type(i)[-1]) for i in range(len(element_types_unique))]
        #     C_nc_cols = max(n_cols_temp)
        #     # ------------------------------------------------------------------
        #     # Loading the part fo the file where the cells connectivity matrix
        #     # is given.
        #     # ------------------------------------------------------------------
        #     C_nc_txt = np.array(c[i_line : i_line + N_size_rows + 1])
        #     # ------------------------------------------------------------------
        #     # Initializing the list of local (individual) connectivity matrix.
        #     # ------------------------------------------------------------------
        #     C_nc_loc_list = []
        #     # ------------------------------------------------------------------
        #     # For all types of elements present in the mesh, building the local
        #     # connectivity matrix.
        #     # ------------------------------------------------------------------
        #     for i in range(len(element_types_unique)):
        #         n_cols = n_cols_temp[i]
        #         # --------------------------------------------------------------
        #         # Getting the position list for the ith element type present in
        #         # the mesh.
        #         # --------------------------------------------------------------
        #         pos = (np.argwhere(element_types == element_types_unique[i])).T[0]
        #         nodes_cols = tuple(range(2, n_cols + 2))
        #         # --------------------------------------------------------------
        #         # Extracting all the local matrices for the ith element type
        #         # present in the mesh.
        #         # --------------------------------------------------------------
        #         C_nc_loc = np.loadtxt(C_nc_txt[pos], usecols=nodes_cols, dtype=int)
        #         # --------------------------------------------------------------
        #         # Redindexing so that the first node has index 0.
        #         # --------------------------------------------------------------
        #         C_nc_loc = C_nc_loc - np.ones(C_nc_loc.shape, dtype=int)
        #         # --------------------------------------------------------------
        #         # Filling voids with -1, so that the gloabl connectivity matrix
        #         # is rectangular.
        #         # --------------------------------------------------------------
        #         C_nc_nan = np.full((len(pos), C_nc_cols - n_cols), -1, dtype=int)
        #         C_nc_loc = np.concatenate((C_nc_loc, C_nc_nan), axis=1)
        #         # --------------------------------------------------------------
        #         # Appenfing to the gloabl connectivity matrix
        #         # --------------------------------------------------------------
        #         C_nc_loc_list.append(C_nc_loc)
        #     C_nc = np.concatenate(C_nc_loc_list, axis=0)
        # # ----------------------------------------------------------------------
        # # if all elements are of the same nature :
        # # ----------------------------------------------------------------------
        # else:
        #     # ------------------------------------------------------------------
        #     # Reasding the only element type.
        #     # ------------------------------------------------------------------
        #     eltype = element_type(0)
        #     C_nc_cols = int(eltype[-1])
        #     nodes_cols = tuple(range(2, C_nc_cols + 2))
        #     # ------------------------------------------------------------------
        #     # Loading the global cell connectivity matrix
        #     # ------------------------------------------------------------------
        #     C_nc = np.loadtxt(c[i_line : i_line + N_size_rows], usecols=nodes_cols, dtype=int,)
        #     C_nc = C_nc - np.ones(C_nc.shape, dtype=int)
        # ######################################################################
        # CREATING FACES CONNECTIVITY MATRIX
        # ######################################################################
        # ----------------------------------------------------------------------
        # Getting the maximum number of points per face in the given mesh,
        # depending on the element type.
        # ----------------------------------------------------------------------
        N_nf_max = max([C_cf_ref[element_type(i)].shape[1] for i in range(len(element_types_unique))])
        # ----------------------------------------------------------------------
        # Getting the maximum number of face per cell in the given mesh, also
        # depending on the element type.
        # ----------------------------------------------------------------------
        N_cf_max = max([C_cf_ref[element_type(i)].shape[0] for i in range(len(element_types_unique))])
        # ----------------------------------------------------------------------
        # Initializing
        # - the faces connectivity matrix C_nf.
        # - the cell-face connectivity matrix C_cf.
        # - a tags list that stores a serial ID for each face, in order not to
        #  add the same face twice in the face connectivity matrix.
        # - a flags vector (with size the number fo faces in the mesh) to store
        # the Nsets each face belongs to.
        # - a weights vector with the weight of each node (the number of cells
        # it belongs to)
        # ----------------------------------------------------------------------
        cell_types = []
        for cell_type in cells_type_matrix:
            if not cell_type in cell_types:
                cell_types.append(cell_type)
        # weights = np.zeros(len(vertices), dtype=int)

        weights = [0 for i in range(len(vertices))]
        tags = []
        flags = []
        cells_faces_adjacency_matrix = []
        faces_vertices_adjacency_matrix = []
        for i in range(len(cells_vertices_adjacency_matrix)):
            for vertex_index in cells_vertices_adjacency_matrix[i]
                weights[vertex_index] += 1
            cell_face_adjacency_matrix = []
            cell_type = cells_type_matrix[i]
            for j in range(len(C_cf_ref[cell_type])):
                face_vertices_adjacency_matrix = []
                for k in range(len(C_cf_ref[cell_type][j])):
                    face_vertices_adjacency_matrix.append(cells_vertices_adjacency_matrix[i][C_cf_ref[cell_type][j]][k])
                # face_vertices_adjacency_matrix = [cells_vertices_adjacency_matrix[i][C_cf_ref[cell_type][j]][k] for k in range(len(C_cf_ref[cell_type][j]))]
                # face_vertices_adjacency_matrix = cells_vertices_adjacency_matrix[i][C_cf_ref[cell_type][j]]
                # C_nc[i][C_cf_ref[eltype][j]]
                tag = "".join([str(item) for item in np.sort(face_vertices_adjacency_matrix)])
                if not tag in tags:
                    # ----------------------------------------------------------
                    # Store it in the face connectivity matrix.
                    # ----------------------------------------------------------
                    flags_local = []
                    faces_vertices_adjacency_matrix.append(face_vertices_adjacency_matrix)
                    # C_nf.append(N_f)
                    tags.append(tag)
                    cell_face_adjacency_matrix.append(len(tags) - 1)
                    # C_cf_loc.append(len(tags) - 1)
                    for key in Nsets.keys():
                        count = 0
                        for vertex_index in face_vertices_adjacency_matrix:
                            if vertex_index in Nsets[key]:
                                count += 1
                        if count == len(face_vertices_adjacency_matrix):
                            flags_local.append(key)
                    if not flags_local:
                        flags_local = ["NONE"]
                    flags.append(flags_local)
                else:
                    cell_face_adjacency_matrix.append(tags.index(tag))
                cells_faces_adjacency_matrix.append(cell_face_adjacency_matrix)


                    #     if np.count_nonzero(np.isin(face_vertices_adjacency_matrix, Nsets[key])) == len(face_vertices_adjacency_matrix):
                    #         flags_local.append(key)

                    # if not flags_local:
                    #     flags_local = ["NONE"]
                    # flags.append(flags_local)


        C_nf = []
        C_cf = []
        tags = []
        flags = []
        weights = np.zeros(N.shape[0], dtype=int)
        # ----------------------------------------------------------------------
        # For each cell :
        # ----------------------------------------------------------------------
        for i in range(len(C_nc)):
            for node_i in C_nc[i]:
                if not node_i == -1:
                    weights[node_i] += 1
            C_cf_loc = []
            eltype = (np.sort(element_types))[i][2:-1]
            # ------------------------------------------------------------------
            # For each face in the ith cell :
            # ------------------------------------------------------------------
            for j in range(len(C_cf_ref[eltype])):
                # --------------------------------------------------------------
                # Extracting the face connectivity matrix.
                # --------------------------------------------------------------
                N_f_init = C_nc[i][C_cf_ref[eltype][j]]
                N_f_nan = np.full((N_nf_max - len(N_f_init)), -1, dtype=int)
                N_f = np.concatenate((N_f_init, N_f_nan))
                tag = "".join([str(item) for item in np.sort(N_f)])
                # --------------------------------------------------------------
                # If the face is not stored yet :
                # --------------------------------------------------------------
                if not tag in tags:
                    # ----------------------------------------------------------
                    # Store it in the face connectivity matrix.
                    # ----------------------------------------------------------
                    flags_local = []
                    C_nf.append(N_f)
                    tags.append(tag)
                    C_cf_loc.append(len(tags) - 1)
                    # ----------------------------------------------------------
                    # For all Nsets, check whether the face belongs to it or
                    # not. If it belongs to any Nset, append flag with the name
                    # of the Nset, and append "NONE" otherwise.
                    # ----------------------------------------------------------
                    for key in Nsets.keys():
                        if np.count_nonzero(np.isin(N_f_init, Nsets[key])) == len(N_f_init):
                            flags_local.append(key)

                    if not flags_local:
                        flags_local = ["NONE"]
                    flags.append(flags_local)
                # --------------------------------------------------------------
                # If the face is already stored :
                # --------------------------------------------------------------
                else:
                    C_cf_loc.append(tags.index(tag))
            # ------------------------------------------------------------------
            # Filling voids with -1 in the local connectivity matrix, so that
            # the gloabl connectivity matrix is rectangular.
            # ------------------------------------------------------------------
            C_cf_loc = np.concatenate(
                (np.array(C_cf_loc), np.full((N_cf_max - len(C_cf_loc)), -1, dtype=int),), axis=0,
            )
            # ------------------------------------------------------------------
            # Appenfing to the gloabl connectivity matrix
            # ------------------------------------------------------------------
            C_cf.append(C_cf_loc)
        C_nf = np.array(C_nf)
        C_cf = np.array(C_cf)
        flags = np.array(flags)
        return N, C_nc, C_nf, C_cf, weights, Nsets, flags
