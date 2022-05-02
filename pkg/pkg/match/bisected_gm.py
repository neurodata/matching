import numpy as np
from numba import jit
from ..match import BaseMatchSolver
from scipy.optimize import linear_sum_assignment


class BisectedGraphMatchSolver(BaseMatchSolver):
    def __init__(
        self,
        A,
        B,
        AB=None,
        BA=None,
        similarity=None,
        partial_match=None,
        rng=None,
        init="barycenter",
        verbose=False,
        shuffle_input=True,
        maximize=True,
        maxiter=30,
        tol=0.01,
    ):
        # TODO more input checking
        super().__init__(
            rng=rng,
            init=init,
            verbose=verbose,
            shuffle_input=shuffle_input,
            maximize=maximize,
            maxiter=maxiter,
            tol=tol,
        )
        # TODO input validation
        # TODO seeds
        # A, B, partial_match = _common_input_validation(A, B, partial_match)

        # TODO similarity
        # if S is None:
        #     S = np.zeros((A.shape[0], B.shape[1]))
        # S = np.atleast_2d(S)

        # TODO padding

        if init == "barycenter":
            init = 1.0

        n = len(B[0])
        nonseed_B = np.setdiff1d(range(n), partial_match[:, 1])

        self.A = _check_input_matrix(A)
        self.B = _check_input_matrix(B)
        self.AB = _check_input_matrix(AB)
        self.BA = _check_input_matrix(BA)

        # self.S = S
        # self.partial_match = partial_match

        # self.n = .shape[0]  # number of vertices in graphs
        # self.n_seeds = partial_match.shape[0]  # number of seeds
        self.n_unseed = A[0].shape[0]  # TODO
        # self.n_nodes = A.shape[0]
        self.n_B = B[0].shape[0]

    def permute(self, permutation):
        self.B = _permute_multilayer(self.B, permutation, rows=True, columns=True)
        self.AB = _permute_multilayer(self.AB, permutation, rows=False, columns=True)
        self.BA = _permute_multilayer(self.BA, permutation, rows=True, columns=False)

    # TODO
    def check_outlier_cases(self):
        pass

    # side_perm = self.rng.permutation(self.n_unseed, 2 * self.n_unseed)
    # perm = np.concatenate((np.arange(self.n_unseed), side_perm))
    # TODO
    def set_reference_frame(self):
        if self.shuffle_input:
            perm = self.rng.permutation(self.n_B)

            self._reverse_permutation = np.argsort(perm)

            self.permute(perm)

            # TODO permute seeds and anything else that could be added
        else:
            self._reverse_permutation = np.arange(self.n_unseed)

    def compute_constant_terms(self):
        # only happens with seeds
        pass

    def compute_step_direction(self, P):
        self.print("Computing step direction")
        grad_fp = self.compute_gradient(P)
        Q, permutation = self.solve_assignment(grad_fp)
        return Q, permutation

    def solve_assignment(self, grad_fp):
        self.print("Solving assignment problem")
        # [1] Algorithm 1 Line 4 - get direction Q by solving Eq. 8
        _, permutation = linear_sum_assignment(grad_fp, maximize=self.maximize)
        Q = np.eye(self.n_unseed)[permutation]
        return Q, permutation

    # permutation is here as a dummy for now
    def compute_step_size(self, P, Q, permutation):
        self.print("Computing step size")
        a, b = _compute_coefficients(P, Q, self.A, self.B, self.AB, self.BA)
        if a * self.obj_func_scalar > 0 and 0 <= -b / (2 * a) <= 1:
            alpha = -b / (2 * a)
        else:
            alpha = np.argmin([0, (b + a) * self.obj_func_scalar])
        return alpha

    def compute_gradient(self, P):
        self.print("Computing gradient")
        gradient = _compute_gradient(P, self.A, self.B, self.AB, self.BA)
        return gradient

    def compute_score(*args):
        return 0


def _permute_multilayer(adjacency, permutation, rows=True, columns=True):
    for layer_index in range(len(adjacency)):
        layer = adjacency[layer_index]
        if rows:
            layer = layer[permutation]
        if columns:
            layer = layer[:, permutation]
        adjacency[layer_index] = layer
    return adjacency


def _check_input_matrix(A):
    if isinstance(A, np.ndarray) and (np.ndim(A) == 2):
        A = np.expand_dims(A, axis=0)
        A = A.astype(float)
    if isinstance(A, list):
        A = np.array(A, dtype=float)
    return A


@jit(nopython=True)
def _compute_gradient(P, A, B, AB, BA, const_sum):
    n_layers = A.shape[0]
    grad = np.zeros_like(P)
    for i in range(n_layers):
        grad += (
            A[i] @ P @ B[i].T
            + A[i].T @ P @ B[i]
            + AB[i] @ P.T @ BA[i].T
            + BA[i].T @ P.T @ AB[i]
            + const_sum[i]
        )

    return grad


@jit(nopython=True)
def _compute_coefficients(P, Q, A, B, AB, BA):
    R = P - Q
    # TODO make these "smart" traces like in the scipy code, couldn't hurt
    # though I don't know how much Numba cares

    n_layers = A.shape[0]
    a_cross = 0
    b_cross = 0
    a_intra = 0
    b_intra = 0
    for i in range(n_layers):
        a_cross += np.trace(AB[i].T @ R @ BA[i] @ R)
        b_cross += np.trace(AB[i].T @ R @ BA[i] @ Q) + np.trace(AB[i].T @ Q @ BA[i] @ R)
        a_intra += np.trace(A[i] @ R @ B[i].T @ R.T)
        b_intra += np.trace(A[i] @ Q @ B[i].T @ R.T + A[i] @ R @ B[i].T @ Q.T)

    a = a_cross + a_intra
    b = b_cross + b_intra

    return a, b
