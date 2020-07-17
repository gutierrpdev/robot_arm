from numpy import sin, cos
import numpy as np
from itertools import product, combinations
from functools import partial


class MotionPlanner2D:

    """ Auxiliary methods for handling subsets """
    @staticmethod
    def findsubsets(s, n):
        return [set(i) for i in combinations(s, n)]

    """ Generate parts of S. """
    @staticmethod
    def powerset(s):
        combs = [list(x) for x in (combinations(s, r) for r in range(len(s) + 1))]
        res = []
        for elem in combs:
            res.extend(elem)
        return res

    """Produce a path connecting A to B through the unique shortest path between them,
    assuming that A != -B. Produce NumSteps frames to represent such path. A and B are
    radial angles in radians."""
    @staticmethod
    def unique_shortest_arc(a, b, num_steps):
        if isinstance(a, list):
            a = a[0]
        if isinstance(b, list):
            b = b[0]
        # Adapt angles to range [0, 2*pi)
        a %= 2 * np.pi
        b %= 2 * np.pi

        step_size = (b - a) / num_steps
        if step_size <= 0:
            return [[a] for i in range(num_steps)]
        path = np.arange(a, b + step_size, step_size)
        return [[elem] for elem in path]

    """Produce a path connecting B to -B clockwise in num_steps"""
    @staticmethod
    def to_antipodal_clockwise(b, num_steps):
        if isinstance(b, list):
            b = b[0]
        # Adapt angles to range [0, 2*pi)
        b %= 2 * np.pi

        step_size = -np.pi / num_steps
        path = np.arange(b, b - np.pi + step_size, step_size) % (2 * np.pi)
        return [[elem] for elem in path]

    """Produce path connecting A to -B in num_steps/2 and then -B to B in num_steps/2."""
    @staticmethod
    def a_minus_b_b(a, b, num_steps):
        if isinstance(a, list):
            a = a[0]
        if isinstance(b, list):
            b = b[0]
        # calculate first path from A to -B
        minus_b = (b + np.pi) % (2 * np.pi)
        path1 = MotionPlanner2D.unique_shortest_arc(a, minus_b, int(num_steps / 2))
        path2 = MotionPlanner2D.to_antipodal_clockwise(minus_b, int(num_steps / 2))
        # second path goes from -B to B
        path1.extend(path2)
        return path1

    """Functions associated to partition of unity in S1 x S1"""
    @staticmethod
    def f1(a, c):
        if isinstance(a, list) and isinstance(c, list):
            return (np.around(cos((a[0] - c[0]) / 2), decimals=5)) ** 2
        else:
            return (np.around(cos((a - c) / 2), decimals=5)) ** 2

    @staticmethod
    def f2(b, d):
        if isinstance(b, list) and isinstance(d, list):
            return (np.around(sin((b[0] - d[0]) / 2), decimals=5)) ** 2
        else:
            return (np.around(sin((b - d) / 2), decimals=5)) ** 2

    """Given A, B, initial/final position in given level, apply first valid algorithm.
    Essentially, we go over each W_k (starting with the lowest k) checking whether the corresponding
    psi_k(a, b) is greater than zero. If this is the case, (a, b) belongs to a single W(S, T) in 
    W_k, and since psi_k is defined 'in pieces' as a list of functions over disjoint W(S, T), we go over
    this list in order to find the (unique) phi_st(a, b) > 0, whose associated algorithm we will them apply."""
    def apply_algorithm(self, level, a, b, num_steps):
        # level's partition. each psi_k contains all phi_ST with |S| + |T| = k.
        psi = self.partitions[level]
        # level's algorithms, one for each W_k
        algorithms = self.algorithms[level]
        k = 2
        for psi_k, alg_k in zip(psi, algorithms):
            print("Checking (A, B) in k = ", k, ":")
            # A, B in psi_k
            for phi_st, alg_st in zip(psi_k, alg_k):
                x = phi_st(a, b)
                print("Phi_ST(A, B) = ", x)
                if x > 0:
                    return alg_st(a, b, num_steps)
            k = k + 1

    """Auxiliary functions. Most of these will be stored as partial functions bound to a specific context. 
    For instance, fun_st will be bound to a concrete tuple (f, g, i, j, ip, jp) in order to become a function
    that only takes a pair (a, b) as an argument. In this sense, these are only conceived as 'blueprints' 
    for specific types of functions which depend on a sensible number of fixed parameters."""
    @staticmethod
    def f_max(f, a, c, num):
        return max([fun(a, c) for fun in f[num]])

    @staticmethod
    def fun_st(f, g, i, j, ip, jp, a, b):
        x = MotionPlanner2D.f_max(f, a[:-1], b[:-1], i) * MotionPlanner2D.f_max(g, a[-1], b[-1], j)
        y = MotionPlanner2D.f_max(f, a[:-1], b[:-1], ip) * MotionPlanner2D.f_max(g, a[-1], b[-1], jp)
        return x - y

    @staticmethod
    def fun_st_full_support(f, g, s, t, a, b):
        res = 1
        for (i, j) in product(s, t):
            res = res * MotionPlanner2D.f_max(f, a[:-1], b[:-1], i) * MotionPlanner2D.f_max(g, a[-1], b[-1], j)
        return res

    @staticmethod
    def phi_st(funcs, a, b):
        return max(min([fun(a, b) for fun in funcs]), 0)

    @staticmethod
    def algorithm_st(alg_1, alg_2, a, b, ns):
        return [a + b for a, b in zip(list(alg_1(a[:-1], b[:-1], ns)), list(alg_2(a[-1], b[-1], ns)))]

    """Compute psi_k function as a list of phi_st functions, as well as the associated algorithm
    for each of the W(S, T)'s comprising W_k."""
    @staticmethod
    def compute_psi_k(f, g, k, alg_f, alg_g):
        # STEP 1: Produce the powersets of {1, ..., n} and {1, 2}, where n is the number of elements in
        # f's partition of unity.
        range_f = set(range(len(f)))
        range_g = set(range(len(g)))
        sets_s = MotionPlanner2D.powerset(range_f)
        sets_t = MotionPlanner2D.powerset(range_g)
        # STEP 2: Compute all pairs (S, T) such that |s_k| + |t_k| = k, with S, T != {}.
        s_k = []
        t_k = []
        for S in sets_s:
            for T in sets_t:
                x = len(S)
                y = len(T)
                if len(S) + len(T) == k and len(S) > 0 and len(T) > 0:
                    s_k.append(S)
                    t_k.append(T)

        # STEP 3: Populate psi_k and alg_k with the corresponding phi_st functions.
        psi_k = []
        alg_k = []
        for s, t in zip(s_k, t_k):
            funcs = []
            for (i, j, ip, jp) in product(s, t, range_f, range_g):
                if ip not in s or jp not in t:
                    # bind function to current context.
                    bound_fun_st = partial(MotionPlanner2D.fun_st, f, g, i, j, ip, jp)
                    funcs.append(bound_fun_st)
            if len(funcs) < 1:  # can use anything
                bound_fun_st = partial(MotionPlanner2D.fun_st_full_support, f, g, s, t)
                funcs.append(bound_fun_st)

            # add phi_st for current S, T combination to psi_k
            phi_st = partial(MotionPlanner2D.phi_st, funcs)
            psi_k.append(phi_st)

            # add associated algorithm to psi_k's list.
            # choose first applicable algorithm
            alg_1 = alg_f[min(s)][0]
            alg_2 = alg_g[min(t)][0]

            alg_st = partial(MotionPlanner2D.algorithm_st, alg_1, alg_2)
            alg_k.append(alg_st)
        return psi_k, alg_k

    """Pre compute a given level of functions. That is, the psi_k functions used for selecting algorithms
    for a given number of joints in a robot arm."""
    def add_level(self, level):
        # calculate phi_ST
        f = self.partitions[level - 1]  # partition for last level
        g = self.partitions[0]  # partition for n = 1
        algs = self.algorithms[level - 1]  # algorithms for last level
        algs_1 = self.algorithms[0]  # algorithms for n = 1
        self.partitions.append([])
        self.algorithms.append([])
        for k in range(2, level + 4):
            psi_k, alg_k = self.compute_psi_k(f, g, k, algs, algs_1)
            self.partitions[level].append(psi_k)
            self.algorithms[level].append(alg_k)

    def __init__(self, num_arms):
        self.num_arms = num_arms
        # current partitions of unity. Initialized with partition for n = 1 (S1 x S1).
        self.partitions = [[[self.f1], [self.f2]]]
        # current algorithms for each open set. Initially 2: U1 and U2.
        self.algorithms = [[[self.unique_shortest_arc], [self.a_minus_b_b]]]
        # initialize as many levels as needed for this specific robot arm planner.
        for i in range(1, num_arms):
            self.add_level(i)


if __name__ == "__main__":
    robot = MotionPlanner2D(2)
    print("Path found:", robot.apply_algorithm(1, [np.pi/2, 0], [np.pi, np.pi], 100))
