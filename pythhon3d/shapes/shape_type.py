from numpy import ndarray as Mat


class ShapeType:
    def __init__(self, problem_dimension: int):
        ""
        ""
        self.face_shape_type = self.get_face_shape_type(problem_dimension)
        self.cell_shape_type = self.get_cell_shape_type(problem_dimension)

    def get_face_shape_type(self, problem_dimension):
        ""
        ""
        if problem_dimension == 1:
            face_shape = "POINT"
        elif problem_dimension == 2:
            face_shape = "LINE"
        elif problem_dimension == 3:
            face_shape = "SURFACE"
        return face_shape

    def get_cell_shape_type(self, problem_dimension):
        ""
        ""
        if problem_dimension == 1:
            cell_shape = "LINE"
        elif problem_dimension == 2:
            cell_shape = "SURFACE"
        elif problem_dimension == 3:
            cell_shape = "VOLUME"
        return cell_shape
