import json

import numba
import numpy

from matplotlib import pyplot


@numba.njit()
def are_coefficients_valid(a_1: float, a_2: float, a_3: float, b: float) -> bool:
    if not a_1 ** 3 - 12 * a_2 * b ** 2 > 0:
        return False

    if not a_1 ** 2 * a_2 - 30 * a_3 * b ** 2 > 0:
        return False

    if not 4 * a_2 ** 2 - 9 * a_1 * a_3 > 0:
        return False

    if not 2 * a_2 - 3 * a_1 * a_3 > 0:
        return False

    if not 84 * a_2 ** 2 - 45 * a_1 * a_3 > 0:
        return False

    return True


def coefficients(start_point: float, end_point: float, step_size: float) -> tuple:
    num_points = int((end_point - start_point) / step_size)
    for a_1_i in range(1, num_points):
        for a_2_i in range(1, num_points):
            for a_3_i in range(1, num_points):
                for b_i in range(1, num_points):
                    a_1 = start_point + a_1_i * step_size
                    a_2 = start_point + a_2_i * step_size
                    a_3 = start_point + a_3_i * step_size
                    b = start_point + b_i * step_size

                    if not are_coefficients_valid(a_1, a_2, a_3, b):
                        continue

                    yield a_1, a_2, a_3, b


@numba.njit()
def get_f(a_1, a_2, a_3, b, x_0=0.0, y_0=0.0, num_iter=20000):
    x = [x_0]
    y = [y_0] 
    
    for i in range(num_iter - 1):
        x_new = 1 - a_1 * x[i] ** 2 - a_2 * x[i] ** 4 - a_3 * x[i] ** 6  + b * y[i]
        y_new = b * x[i]
        
        x.append(x_new)
        y.append(y_new)

    return x, y

# @numba.njit()
def calculate_degree(a_1, a_2, a_3, b, x_0=0, y_0=0):
    vec_1 = numpy.array([1, 0])
    vec_2 = numpy.array([0, 1])

    x_array = [x_0]
    y_array = [y_0]

    num_iter = 2000

    for i in range(1, num_iter + 1):
        x = 1 - a_1 * (x_array[i - 1] ** 2) - a_2 * (x_array[i - 1] ** 4) - a_3 * (x_array[i - 1] ** 6)  + b * y_array[i - 1]
        y = b * x_array[i - 1]

        x_array.append(x)
        y_array.append(y)

        jordan_matrix = numpy.array([[-2 * a_1 * x - 4 * a_2 * (x ** 3) - 6 * a_3 * (x ** 5), b], [b, 0]])
        vec_1 = numpy.dot(jordan_matrix, vec_1)
        vec_2 = numpy.dot(jordan_matrix, vec_2) 

        dot_product_1 = numpy.dot(vec_1, vec_1)
        dot_product_2 = numpy.dot(vec_1, vec_2)
        vec_2 = vec_2 - (dot_product_2 / dot_product_1) * vec_1
        length_v_1 = numpy.sqrt(dot_product_1)
        area = numpy.abs(vec_1[0] * vec_2[1] - vec_1[1] * vec_2[0])

        h_1 = numpy.log(length_v_1) / i
        h_2 = numpy.log(area) / i - h_1

    degree = 1 - h_1 / h_2
    
    return degree
