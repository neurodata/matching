import time
from functools import wraps

import numpy as np
from numba import jit
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from graspologic.types import Dict, Tuple


class GraphMatchSolver(BaseEstimator):
    def __init__(
        self,
        A,
        B,
        AB=None,
        BA=None,
        similarity=None,
        partial_match=None,
        rng=None,
        init=1.0,
        verbose=False,
        shuffle_input=True,
        maximize=True,
        maxiter=30,
        tol=0.01,
    ):
        # TODO more input checking
        self.rng = check_random_state(rng)
        self.init = init
        self.verbose = verbose
        self.shuffle_input = shuffle_input
        self.maximize = maximize
        self.maxiter = maxiter
        self.tol = tol

        if maximize:
            self.obj_func_scalar = -1
        else:
            self.obj_func_scalar = 1

        if partial_match is None:
            partial_match = np.array([[], []]).astype(int).T
            self._seeded = False
        else:
            self._seeded = True

        # TODO input validation
        # TODO seeds
        # A, B, partial_match = _common_input_validation(A, B, partial_match)

        # TODO similarity
        # if S is None:
        #     S = np.zeros((A.shape[0], B.shape[1]))
        # S = np.atleast_2d(S)

        # TODO padding

        # TODO make B always bigger

        # convert everything to make sure they are 3D arrays (first dim is layer)
        A = _check_input_matrix(A)
        B = _check_input_matrix(B)
        AB = _check_input_matrix(AB)
        BA = _check_input_matrix(BA)

        if AB is None:
            AB = np.zeros((A.shape[1], B.shape[1]))
        if BA is None:
            BA = np.zeros((B.shape[1], A.shape[1]))

        n_seeds = len(partial_match)

        # set up so that seeds are first and we can grab subgraphs easily
        # TODO could also do this slightly more efficiently just w/ smart indexing?
        nonseed_A = np.setdiff1d(range(len(A[0])), partial_match[:, 0])
        nonseed_B = np.setdiff1d(range(len(B[0])), partial_match[:, 1])
        perm_A = np.concatenate([partial_match[:, 0], nonseed_A])
        perm_B = np.concatenate([partial_match[:, 1], nonseed_B])

        # permute each (sub)graph appropriately
        A = _permute_multilayer(A, perm_A, rows=True, columns=True)
        B = _permute_multilayer(B, perm_B, rows=True, columns=True)
        AB = _permute_multilayer(AB, perm_A, rows=True, columns=False)
        AB = _permute_multilayer(AB, perm_B, rows=False, columns=True)
        BA = _permute_multilayer(BA, perm_A, rows=False, columns=True)
        BA = _permute_multilayer(BA, perm_B, rows=True, columns=False)

        # split into subgraphs of seed-to-seed, seed-to-nonseed, etc.
        # main thing being permuted has no subscript
        self.A_ss, self.A_sn, self.A_ns, self.A = _split_matrix(A, n_seeds)
        self.B_ss, self.B_sn, self.B_ns, self.B = _split_matrix(B, n_seeds)
        self.AB_ss, self.AB_sn, self.AB_ns, self.AB = _split_matrix(AB, n_seeds)
        self.BA_ss, self.BA_sn, self.BA_ns, self.BA = _split_matrix(BA, n_seeds)

        self.n_unseed = self.A[0].shape[0]
        self.n_B = self.B[0].shape[0]

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
    # def set_reference_frame(self):
    #     if self.shuffle_input:
    #         perm = self.rng.permutation(self.n_B)

    #         self._reverse_permutation = np.argsort(perm)

    #         self.permute(perm)

    #         # TODO permute seeds and anything else that could be added
    #     else:
    #         self._reverse_permutation = np.arange(self.n_unseed)

    def compute_constant_terms(self):
        if self._seeded:
            n_layers = len(self.A)
            ipsi = []
            contra = []
            for i in range(n_layers):
                ipsi.append(
                    self.A_ns[i] @ self.B_ns[i].T + self.A_sn[i].T @ self.B_sn[i]
                )
                contra.append(
                    self.AB_ns[i] @ self.BA_ns[i].T + self.BA_sn[i].T @ self.AB_sn[i]
                )
            ipsi = np.array(ipsi)
            contra = np.array(contra)
            self.ipsi_constant_sum = ipsi
            self.contra_constant_sum = contra
            self.constant_sum = ipsi + contra
        else:
            self.constant_sum = np.zeros(self.B.shape)

    def compute_step_direction(self, P):
        self.print("Computing step direction")
        grad_fp = self.compute_gradient(P)
        Q = self.solve_assignment(grad_fp)
        return Q

    def solve_assignment(self, grad_fp):
        self.print("Solving assignment problem")
        # [1] Algorithm 1 Line 4 - get direction Q by solving Eq. 8
        _, permutation = linear_sum_assignment(grad_fp, maximize=self.maximize)
        Q = np.eye(self.n_unseed)[permutation]
        return Q

    # permutation is here as a dummy for now
    def compute_step_size(self, P, Q):
        self.print("Computing step size")
        a, b = _compute_coefficients(
            P,
            Q,
            self.A,
            self.B,
            self.AB,
            self.BA,
        )
        if a * self.obj_func_scalar > 0 and 0 <= -b / (2 * a) <= 1:
            alpha = -b / (2 * a)
        else:
            alpha = np.argmin([0, (b + a) * self.obj_func_scalar])
        return alpha

    def compute_gradient(self, P):
        self.print("Computing gradient")
        gradient = _compute_gradient(
            P, self.A, self.B, self.AB, self.BA, self.constant_sum
        )
        return gradient

    def compute_score(*args):
        return 0

    def status(self):
        if hasattr(self, "n_iter"):
            return f"[Iteration: {self.n_iter}]"
        else:
            return "[Pre-loop]"

    def print(self, msg):
        if self.verbose:
            status = self.status()
            print(status + " " + msg)

    def check_converged(self, P, P_new):
        return np.linalg.norm(P - P_new) / np.sqrt(self.n_unseed) < self.tol

    def solve(self):
        self.check_outlier_cases()
        # self.set_reference_frame()

        P = self.initialize()
        self.compute_constant_terms()
        for n_iter in range(self.maxiter):
            self.n_iter = n_iter

            Q = self.compute_step_direction(P)
            alpha = self.compute_step_size(P, Q)

            # take a step in this direction
            P_new = alpha * P + (1 - alpha) * Q

            if self.check_converged(P, P_new):
                self.converged = True
                break
            P = P_new

        self.finalize(P)

    def initialize(self):
        self.print("Initializing")
        if isinstance(self.init, float):
            n_unseed = self.n_unseed
            rng = self.rng
            J = np.ones((n_unseed, n_unseed)) / n_unseed
            # DO linear combo from barycenter
            K = rng.uniform(size=(n_unseed, n_unseed))
            # Sinkhorn balancing
            K = _doubly_stochastic(K)
            P = J * self.init + K * (1 - self.init)  # TODO check how defined in paper
        elif isinstance(self.init, np.ndarray):
            raise NotImplementedError()
            # TODO fix below
            # P0 = np.atleast_2d(P0)
            # _check_init_input(P0, n_unseed)
            # invert_inds = np.argsort(nonseed_B)
            # perm_nonseed_B = np.argsort(invert_inds)
            # P = P0[:, perm_nonseed_B]

        self.converged = False
        return P

    def finalize(self, P):
        self.print("Finalizing permutation")
        # P = self.unset_reference_frame(P)
        self.P_final_ = P

        _, permutation = linear_sum_assignment(P, maximize=self.maximize)
        self.permutation_ = permutation

        score = self.compute_score(permutation)
        self.score_ = score

    # def unset_reference_frame(self, P):
    #     reverse_perm = self._reverse_permutation
    #     P = P[:, reverse_perm]
    #     return P


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


# A_ns, A_sn, B_ns, B_sn, AB_ns, AB_sn,


@jit(nopython=True)
def _compute_coefficients(
    P,
    Q,
    A,
    B,
    AB,
    BA,
):
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


def timer(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        sec = te - ts
        output = f"Function {f.__name__} took {sec:.3f} seconds."
        print(output)
        return result

    return wrap


# REF: https://github.com/microsoft/graspologic/blob/dev/graspologic/match/qap.py
def _doubly_stochastic(P: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    # Adapted from @btaba implementation
    # https://github.com/btaba/sinkhorn_knopp
    # of Sinkhorn-Knopp algorithm
    # https://projecteuclid.org/euclid.pjm/1102992505

    max_iter = 1000
    c = 1 / P.sum(axis=0)
    r = 1 / (P @ c)
    P_eps = P

    for it in range(max_iter):
        if (np.abs(P_eps.sum(axis=1) - 1) < tol).all() and (
            np.abs(P_eps.sum(axis=0) - 1) < tol
        ).all():
            # All column/row sums ~= 1 within threshold
            break

        c = 1 / (r @ P)
        r = 1 / (P @ c)
        P_eps = r[:, None] * P * c

    return P_eps


def _split_matrix(
    matrices: np.ndarray, n: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # definitions according to Seeded Graph Matching [2].
    n_layers = matrices.shape[0]
    seed_to_seed = []
    seed_to_nonseed = []
    nonseed_to_seed = []
    nonseed_to_nonseed = []
    for i in range(n_layers):
        X = matrices[i]
        upper, lower = X[:n], X[n:]
        seed_to_seed.append(upper[:, :n])
        seed_to_nonseed.append(upper[:, n:])
        nonseed_to_seed.append(lower[:, :n])
        nonseed_to_nonseed.append(lower[:, n:])
    seed_to_seed = np.array(seed_to_seed)
    seed_to_nonseed = np.array(seed_to_nonseed)
    nonseed_to_seed = np.array(nonseed_to_seed)
    nonseed_to_nonseed = np.array(nonseed_to_nonseed)
    return seed_to_seed, seed_to_nonseed, nonseed_to_seed, nonseed_to_nonseed
