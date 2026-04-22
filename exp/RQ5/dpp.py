import numpy as np
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpContinuous


def dpp(c, nt, d_ct_matrix, lp='linear'):
    """
    Discrete or Linear Programming for constructing M_ct matrix.

    Parameters:
    -----------
    c : int
        Number of shared classes between source and target.
    nt : int
        Number of target domain samples.
    d_ct_matrix : ndarray
        Matrix of d_ct values, shape (nt, c).
    lp : str, optional
        Type of linear programming: 'linear' (default) for continuous [0,1],
        'binary' for binary {0,1} variables.

    Returns:
    --------
    m_ct_matrix : ndarray
        Matrix of M_ct values, shape (nt, c).
    """

    # total variables: x_0, x_1, ..., x_{nt*c-1}
    num_vars = nt * c

    # create LP problem
    prob = LpProblem("DPP_Problem", LpMinimize)

    if lp == 'binary':
        var_type = LpBinary
    elif lp == 'linear':
        var_type = LpContinuous
    else:
        raise ValueError("lp must be either 'linear' or 'binary'")

    x_vars = [LpVariable(f'x_{i}', lowBound=0, upBound=1, cat=var_type) for i in range(num_vars)]

    # objective function: Minimize the sum of (D_ct[i, j] * x_{i*C + j})
    d_vec = d_ct_matrix.reshape(-1)  # convert to vector(nt * c, )
    objective = lpSum([d_vec[i] * x_vars[i] for i in range(num_vars)])
    prob += objective

    # equality constraints: The sum of x_ct for each sample (row) is equal to 1.
    # That is, for each t in 1...nt: sum_c x_{(t - 1)*c + c} = 1
    for t in range(nt):
        # variable index range: (t * c) to ((t + 1) * c) - 1
        indices = [t * c + j for j in range(c)]
        prob += lpSum([x_vars[i] for i in indices]) == 1

    # inequality constraint: For each class c, sum_t x_{(t-1)*c + (c-1)} <= 1
    for class_idx in range(c):
        # Variable index: t ∈ 0...nt-1. The class_idx-th variable for each t corresponds to t * c + class_idx.
        indices = [t * c + class_idx for t in range(nt)]
        prob += lpSum([x_vars[i] for i in indices]) <= 1

    prob.solve()

    solution = np.zeros(num_vars)
    for i in range(num_vars):
        solution[i] = x_vars[i].varValue

    m_ct_vector = solution  # shape: (nt * c, )
    m_ct_matrix = m_ct_vector.reshape((nt, c))  # shape: (nt, c)

    return m_ct_matrix
