import math
from typing import (
    List,
    Optional,
    Tuple,
    )

# matplotlib.use('tkagg')
import numpy as np
import scipy

from ..data_logging.data_recorder import DataRecorder
from .ask_tell_optimizer import AskTellOptimizer
from .dimension.gaussian_dimension import DimensionInfo, Gaussian


# sys.path.append('../examples/flatirons')


class CMAESOptimizer(AskTellOptimizer):
    """
    An implementation of the covariance matrix adaptation evolution strategy
    
    http://www.cmap.polytechnique.fr/~nikolaus.hansen/copenhagen-cma-es.pdf
    https://arxiv.org/pdf/1604.00772.pdf
    """
    
    recorder: Optional[DataRecorder]
    _best_candidate: Optional[Tuple[float, float, any]]
    verbose: bool
    
    _lambda: int
    mu: int
    c_m: float
    sigma: float
    
    n: int
    chi_n: float
    C: np.ndarray
    C: np.ndarray
    p_c: np.ndarray
    p_sigma: np.ndarray
    
    c_c: float
    c_sigma: float
    c_1: float
    
    w: np.ndarray
    mu_eff: float
    sum_w: float
    c_mu: float
    d_sigma: float
    
    m: np.ndarray
    D: np.ndarray
    B: np.ndarray
    eigenvalue: float
    count_eval: int
    
    def __init__(self,
                 generation_size: int = 100,
                 selection_proportion: float = .5,
                 c_m: float = 1.0,
                 sigma: float = .5,
                 dimensions: Optional[List[DimensionInfo]] = None,
                 verbose: bool = False,
                 ) -> None:
        self.recorder = None
        self._best_candidate = None
        self.verbose = verbose
        
        self._lambda = generation_size
        self.mu = math.floor(selection_proportion * generation_size)
        self.c_m = c_m
        self.sigma = sigma
        self.alpha_cov = 2
        
        if dimensions is not None:
            self.setup(dimensions, self.recorder)
    
    def setup(self, dimensions: [Gaussian], recorder: DataRecorder) -> None:
        """
        Setup parameters given initial conditions of the candidate
        :param dimensions: list of search dimensions
        :param recorder: data recorder
        """
        n = len(dimensions)
        self.n = n
        self.print('_n {}', self.n)
        
        self.chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * (n ** 2)))
        self.print('_chi_n {}', self.chi_n)
        
        self.m = np.fromiter((d.mean() for d in dimensions), float).reshape((n, 1))
        self.print('_m {}', self.m)
        
        # Strategy parameter setting: Selection
        
        # muXone recombination weights
        w = np.zeros((self._lambda, 1))
        for i in range(self.mu):
            w[i] = math.log(self.mu + .5) - math.log(i + 1)
        w /= np.sum(w)  # normalize recombination weights array
        self.w = w
        self.print('_w {}', self.w)
        self.print('sum w_i, i = 1 ... mu {}', sum([w[i] for i in range(self.mu)]))
        
        #  variance-effective size of mu
        self.mu_eff = np.sum(w) ** 2 / np.sum(w ** 2)
        self.print('_mu_eff {}', self.mu_eff)
        
        # Strategy parameter setting: Adaptation
        
        # time constant for accumulation for C
        self.c_c = (4 + self.mu_eff / n) / (n + 4 + 2 * self.mu_eff / n)
        self.print('_c_c {}', self.c_c)
        
        # t-const for cumulation for sigma control
        self.c_sigma = (self.mu_eff + 2) / (n + self.mu_eff + 5)
        self.print('_c_sigma {}', self.c_sigma)
        
        # learning rate for rank-one update of C
        # self.c_1 = self.alpha_cov / ((n + 1.3) ** 2 + self.mu_eff)
        self.c_1 = 2 / ((n + 1.3) ** 2 + self.mu_eff)
        self.print('_c_1 {}', self.c_1)
        
        # learning rate for rank-mu update
        # self.c_mu = min(1 - self.c_1,
        #                 self.alpha_cov * (self.mu_eff - 2 + 1.0 / self.mu_eff) /
        #                 ((n + 2) ** 2 + self.alpha_cov * self.mu_eff / 2))
        self.c_mu = 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((n + 2) ** 2 + 2 * self.mu_eff / 2)
        self.print('_c_mu {}', self.c_mu)
        
        #  damping for sigma
        self.d_sigma = 1 + 2 * max(0.0, math.sqrt((self.mu_eff - 1) / (n + 1)) - 1) + self.c_sigma
        self.print('_d_sigma {}', self.d_sigma)
        
        # Initialize dynamic (internal) strategy parameters and constants
        self.p_c = np.zeros((n, 1), dtype=np.float64)
        self.p_sigma = np.zeros((n, 1), dtype=np.float64)
        self.B = np.eye(n)
        self.D = np.eye(n)
        
        for i, d in enumerate(dimensions):
            self.D[i, i] = math.sqrt(d.variance())
        self.print('D\n{}', self.D)
        
        BD = np.matmul(self.B, self.D)
        self.C = np.matmul(BD, BD.transpose())
        
        self.print('C\n{}', self.C)
        
        self.eigenvalue = 0.0
        
        self.count_eval = 0
        
        self.recorder = recorder
        self.recorder.add_columns('generation', 'mean', 'variance', 'covariance', '_sigma', '_p_c', '_p_sigma')
    
    def stop(self) -> bool:
        """
        :return: True when the optimizer thinks it has reached a stopping point
        """
        return False
    
    def ask(self, num: Optional[int] = None) -> [any]:
        """

        :param num: the number of search points to return. If undefined, the optimizer will choose how many to return.
        :return: a list of search points generated by the optimizer
        """
        n = self.n
        zero = np.zeros(n)
        eye = np.eye(n)
        BD = np.matmul(self.B, self.D)
        # self.print('B\n{}', self.B)
        # self.print('D\n{}', self.D)
        # self.print('BD\n{}', BD)
        
        candidates = [self.m.reshape(n)]
        for _ in range(self._lambda):
            z = np.random.multivariate_normal(zero, eye).reshape((n, 1))
            # self.print('z\n{}', z)
            x = self.m + self.sigma * np.matmul(BD, z)
            # self.print('x\n{}', x)
            candidates.append(x.reshape(n))
        self.count_eval += len(candidates)
        
        return candidates
    
    def tell(self, evaluations: [Tuple[float, float, any]]) -> None:
        """
        Updates the optimizer with the objective evaluations of a list of search points
        :param evaluations: a list of tuples of (evaluation, search point)
        """

        def best_key(e):
            return e[1], e[0]

        best = max(evaluations, key=best_key)
        self._best_candidate = best if self._best_candidate is None else max((self._best_candidate, best), key=best_key)
        
        evaluations.sort(key=lambda e: (e[0], e[1]), reverse=True)
        
        # selection and recombination
        x = [np.array(e[2]).reshape((self.n, 1)) for e in evaluations]
        # self.print('x\n{}', x)
        y = [(x_i - self.m) / self.sigma for x_i in x]
        
        BD = np.matmul(self.B, self.D)
        # self.print('B\n{}', self.B)
        # self.print('D\n{}', self.D)
        # self.print('BD\n{}', BD)
        BDinv = np.linalg.inv(BD)
        # self.print('BDinv\n{}', BDinv)
        # C = np.matmul(BD, BD.transpose())
        # Cinv = np.matmul(np.linalg.inv(self.D), self.B.transpose())
        z = [np.matmul(BDinv, y_i) for y_i in y]
        
        self.m = sum((self.w[i] * x[i] for i in range(self._lambda)))
        z_mean = sum((self.w[i] * z[i] for i in range(self._lambda)))
        
        # Accumulation: Update evolution paths
        
        self.print('m\n{}', self.m)
        self.print('z_mean\n{}', z_mean)
        self.print('B\n{}', self.B)
        
        self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + \
                       math.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * \
                       np.matmul(self.B, z_mean)
        self.print('p_sigma {}', self.p_sigma)
        
        h_sigma = 0.0
        h_sigma_threshold = np.linalg.norm(self.p_sigma) / np.sqrt(
            1 - (1 - self.c_sigma) ** (2 * self.count_eval / self._lambda)) / self.chi_n
        self.print('h_sigma_threshold {}', h_sigma_threshold)
        if h_sigma_threshold < (1.4 + 2 / (self.n + 1)):
            h_sigma = 1
        self.print('h_sigma {}', h_sigma)
        
        # self.print('pc1\n{}', (1.0 - self.c_c) * self.p_c)
        # self.print('pc2\n{}', math.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff))
        # self.print('pc3\n{}', h_sigma * math.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff))
        # self.print('pc4\n{}', z_mean)
        # self.print('pc5\n{}', BD)
        # self.print('pc6\n{}', np.matmul(BD, z_mean))
        self.p_c = (1.0 - self.c_c) * self.p_c + \
                   h_sigma * math.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * np.matmul(BD, z_mean)
        self.print('p_c\n{}', self.p_c)
        
        # Adapt covariance matrix C
        # self.print('C1\n{}', self.C)
        self.print('C2 {}', (1 - self.c_1 - self.c_mu))
        # self.print('C2\n{}', (1 - self.c_1 - self.c_mu) * self.C)
        # self.print('C3\n{}', self.p_c)
        # self.print('C4\n{}', np.matmul(self.p_c, self.p_c.transpose()))
        # self.print('C6 {}', (1 - h_sigma) * self.c_c * (2 - self.c_c))
        self.print('c_1 {}', self.c_1)
        self.print('c_mu {}', self.c_mu)
        # self.print('C7\n{}', (1 - h_sigma) * self.c_c * (2 - self.c_c) * self.C)
        # print('C5\n', self.c_1 * (
        #         np.matmul(self.p_c, self.p_c.transpose()) + (1 - h_sigma) * self.c_c * (2 - self.c_c) * self.C))
        # self.print('C8\n{}', sum((self.w[i] * np.matmul(y[i], y[i].transpose()) for i in range(self._lambda))))
        # self.print('C9\n{}',
        #            self.c_mu * sum((self.w[i] * np.matmul(y[i], y[i].transpose()) for i in range(self._lambda))))
        self.C = (1 - self.c_1 - self.c_mu) * self.C + \
                 self.c_1 * (np.matmul(self.p_c, self.p_c.transpose()) +
                             (1 - h_sigma) * self.c_c * (2 - self.c_c) * self.C) + \
                 self.c_mu * sum((self.w[i] * np.matmul(y[i], y[i].transpose()) for i in range(self._lambda)))
        self.print('C\n{}', self.C)
        
        # Adapt step-size sigma
        p_sigma_norm = scipy.linalg.norm(self.p_sigma)
        self.print('p_sigma_norm {}', p_sigma_norm)
        
        self.sigma = self.sigma * math.exp(
            (self.c_sigma / self.d_sigma) *
            (p_sigma_norm / self.chi_n - 1))
        self.print('sigma {}', self.sigma)
        
        # Update B and D from C
        if self.count_eval - self.eigenvalue > self._lambda / (self.c_1 + self.c_mu) / self.n / 10:
            self.eigenvalue = self.count_eval
            self.print('eigenvalue {}', self.eigenvalue)
            
            self.C = np.real(np.triu(self.C) + np.triu(self.C, 1).transpose())  # force symmetry
            # self.print('C symmetric\n{}', self.C)
            
            w, v = np.linalg.eig(self.C)  # eigen decomposition
            
            self.D = np.real(np.diag(w))
            self.B = np.real(v)
            # self.B, self.D
            # B are the normalized eigenvectors
            
            # self.print('D initial\n{}', self.D)
            
            self.D = np.real(np.diag(np.sqrt(np.diag(self.D))))
            # D are the standard deviations
            
            # self.print('B\n{}', self.B)
            # self.print('D\n{}', self.D)
        
        # Escape flat fitness
        if evaluations[0][0] == evaluations[math.ceil(.7 * self._lambda)][0]:
            self.sigma *= math.exp(.2 + self.c_sigma / self.d_sigma)
            print('warning: flat fitness, consider reformulating the objective')
        
        self.recorder.accumulate(evaluations, self.mean(), self.variance(), self.C, self.sigma, self.p_c,
                                 self.p_sigma)
    
    def best_solution(self) -> Optional[Tuple[float, float, any]]:
        
        """
        :return: the current best solution
        """
        return self._best_candidate
    
    def central_solution(self) -> (Optional[float], Optional[float], any):
        return None, None, self.mean()
    
    def get_num_candidates(self) -> int:
        return self._lambda
    
    def get_num_dimensions(self) -> int:
        return self.m.size
    
    def mean(self) -> any:
        return self.m.reshape(self.get_num_dimensions())
    
    def variance(self) -> any:
        return self.C.diagonal()
    
    def print(self, message: str, *args, **kwargs) -> None:
        if self.verbose:
            print(message.format(*args, **kwargs))
    
    # def setup(self, dimensions: [Gaussian], recorder: DataRecorder) -> None:
    #     """
    #     Setup parameters given initial conditions of the candidate
    #     :param dimensions: list of search dimensions
    #     :param recorder: data recorder
    #     """
    #     n = len(dimensions)
    #     self.n = n
    #     self.print('_n {}', self.n)
    #
    #     self.chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * (n ** 2)))
    #     self.print('_chi_n {}', self.chi_n)
    #
    #     self.p_sigma = np.zeros(n, dtype=np.float64)
    #     self.p_c = np.zeros((n, n), dtype=np.float64)
    #
    #     self.m = np.fromiter((d.mean() for d in dimensions), float)
    #     self.print('_m {}', self.m)
    #     self.C = np.zeros((n, n))
    #     for i, d in enumerate(dimensions):
    #         self.C[i, i] = d.variance()
    #     self.print('_C {}', self.C)
    #
    #     # self.w = np.zeros(self._lambda)
    #
    #     #._lambda, w, c_sigma, d_sigma, c_c, c_1, c_mu
    #     w_prime = np.zeros(self._lambda)
    #     weight = math.log((self._lambda + 1) / 2)
    #     for i in range(self._lambda):
    #         w_prime[i] = weight - math.log(i + 1)
    #     self.print('w_prime {}', w_prime)
    #
    #     self.mu_eff = np.sum(w_prime) ** 2 / np.sum(w_prime ** 2)
    #     self.print('_mu_eff {}', self.mu_eff)
    #
    #     mu_mask = np.zeros(self._lambda)
    #     for i in range(self.mu):
    #         mu_mask[i] = 1.0
    #
    #     w_prime_mu = w_prime * mu_mask
    #     mu_eff_neg = np.sum(w_prime_mu) ** 2 / np.sum(w_prime_mu ** 2)
    #     self.print('mu_eff_bar {}', mu_eff_neg)
    #
    #     self.c_1 = self.alpha_cov / ((n + 1.3) ** 2 + self.mu_eff)
    #     self.print('_c_1 {}', self.c_1)
    #
    #     self.c_c = (4 + self.mu_eff / n) / (n + 4 + 2 * self.mu_eff / n)
    #     self.print('_c_c {}', self.c_c)
    #
    #     self.c_mu = min(1 - self.c_1,
    #                      self.alpha_cov * (self.mu_eff - 2 + 1.0 / self.mu_eff) /
    #                      ((n + 2) ** 2 + self.alpha_cov * self.mu_eff / 2))
    #     self.print('_c_mu {}', self.c_mu)
    #
    #     self.c_sigma = (self.mu_eff + 2) / (n + self.mu_eff + 5)
    #     self.print('_c_sigma {}', self.c_sigma)
    #
    #     self.d_sigma = 1 + 2 * max(0.0, math.sqrt((self.mu_eff - 1) / (n + 1)) - 1) + self.c_sigma
    #     self.print('_d_sigma {}', self.d_sigma)
    #
    #     alpha_mu_neg = 1 + self.c_1 / self.c_mu
    #     self.print('alpha_mu_neg {}', alpha_mu_neg)
    #
    #     alpha_mu_eff_neg = 1 + (2 * mu_eff_neg) / (self.mu_eff + 2)
    #     self.print('alpha_mu_eff_neg {}', alpha_mu_eff_neg)
    #
    #     alpha_pos_def_neg = (1 - self.c_1 - self.c_mu) / (n * self.c_mu)
    #     self.print('alpha_pos_def_neg {}', alpha_pos_def_neg)
    #
    #     min_alpha = min(alpha_mu_neg, alpha_mu_eff_neg, alpha_pos_def_neg)
    #     self.print('min_alpha {}', min_alpha)
    #
    #     sum_positive_w_prime = 0.0
    #     sum_negative_w_prime = 0.0
    #     for w_prime_i in w_prime:
    #         if w_prime_i > 0:
    #             sum_positive_w_prime += w_prime_i
    #         else:
    #             sum_negative_w_prime -= w_prime_i
    #     self.print('sum_positive_w_prime {}', sum_positive_w_prime)
    #     self.print('sum_negative_w_prime {}', sum_negative_w_prime)
    #
    #     w = np.zeros(self._lambda)
    #     for i in range(self._lambda):
    #         w_prime_i = w_prime[i]
    #         w_i = 0.0
    #         if w_prime_i >= 0:
    #             w_i = w_prime_i / sum_positive_w_prime
    #         else:
    #             w_i = w_prime_i * min_alpha / sum_negative_w_prime
    #         w[i] = w_i
    #     self.w = w
    #     self.print('_w {}', self.w)
    #     self.print('sum w_i, i = 1 ... mu {}', sum([w[i] for i in range(self.mu)]))
    #
    #     self.sum_w = np.sum(self.w)
    #     self.print('_sum_w {}', self.sum_w)
    #
    #     # make sure that u_w ~= .3._lambda
    #     # norm_weights = math.sqrt(.3 * self._lambda / np.sum(self.w ** 2))
    #     # self.w *= norm_weights
    #
    #     self.c_m = 1.0
    #     self.print('_c_m {}', self.c_m)
    #
    #     # self.c_mu = min(1.0, self.mu_eff / n ** 2)
    #     # c_norm = max(1.0, self.c_1 + self.c_mu)
    #     # self.c_1 /= c_norm
    #     # self.c_mu /= c_norm
    #     # self.d_sigma = 1 + math.sqrt(self.mu_eff / n)
    #
    #     self.recorder = recorder
    #     self.recorder.add_columns('generation', 'mean', 'variance', 'covariance', '_sigma', '_p_c', '_p_sigma')
    # def tell(self, evaluations: [Tuple[float, any]]) -> None:
    #     """
    #     Updates the optimizer with the objective evaluations of a list of search points
    #     :param evaluations: a list of tuples of (evaluation, search point)
    #     """
    #     evaluations.sort(key._lambda evaluation: evaluation[0], reverse=True)
    #     if self.best_candidate is None or evaluations[0][0] > self.best_candidate[0]:
    #         self.best_candidate = evaluations[0]
    #
    #     # selection_size = math.ceil(self.selection_proportion * len(evaluations))
    #     # del evaluations[selection_size:]
    #
    #     # selection and recombination
    #     x = [e[1] for e in evaluations]
    #     y = [(x_i - self.m) / self.sigma for x_i in x]
    #     y_weighted = [self.w[i] * y_i for i, y_i in enumerate(y)]
    #     y_weighted_sum = sum(y_weighted)
    #     self.print('y_weighted_sum {}', y_weighted_sum)
    #
    #     print('_m', self.m, self.c_m * self.sigma * y_weighted_sum,
    #           self.m + self.c_m * self.sigma * y_weighted_sum)
    #     self.m = self.m + self.c_m * self.sigma * y_weighted_sum
    #
    #     inv_sqrt_C = scipy.linalg.fractional_matrix_power(self.C, -.5)
    #     self.print('inv_sqrt_C {}', y_weighted_sum)
    #
    #     # step-size control
    #     # ps = (1 - cs) * ps...
    #     # + sqrt(cs * (2 - cs) * mueff) * invsqrtC * (xmean - xold) / sigma;
    #     self.p_sigma = (1.0 - self.c_sigma) * self.p_sigma + \
    #                     math.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * \
    #                     np.matmul(inv_sqrt_C, y_weighted_sum)
    #     self.print('_p_sigma {}', self.p_sigma)
    #
    #     # hsig = sum(ps. ^ 2) / (1 - (1 - cs) ^ (2 * counteval ._lambda)) / N < 2 + 4 / (N + 1);
    #     p_sigma_norm = scipy.linalg.norm(self.p_sigma)
    #     self.print('p_sigma_norm {}', p_sigma_norm)
    #
    #     self.sigma = self.sigma * math.exp(
    #         (self.c_sigma / self.d_sigma) *
    #         (p_sigma_norm / self.chi_n - 1))
    #     self.print('_sigma {}', self.sigma)
    #
    #     # covariance matrix adaptation
    #     # hsig = sum(ps. ^ 2) / (1 - (1 - cs) ^ (2 * counteval ._lambda)) / N < 2 + 4 / (N + 1);
    #     h_sigma = 0.0
    #     if p_sigma_norm / np.sqrt(1.0 - (1.0 - self.c_sigma) ** 2) < (1.4 + 2 / (self.n + 1)) * self.chi_n:
    #         h_sigma = 1
    #     self.print('h_sigma {}', h_sigma)
    #
    #     self.p_c = (1.0 - self.c_c) * self.p_c + \
    #                 h_sigma * math.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff) * y_weighted_sum
    #     self.print('_p_c {}', self.p_c)
    #
    #     # w_i_dot = [w_i if w_i >= 0 else self.n / (np.linalg.norm(np.) ** 2) for i, w_i in enumerate(self.w)]
    #
    #     del_h_sigma = 1 if (1 - h_sigma) * self.c_c * (2 - self.c_c) <= 1 else 0
    #     self.print('del_h_sigma {}', del_h_sigma)
    #
    #     w_dot = np.zeros(self.w.shape)
    #     for i, w_i in enumerate(self.w):
    #         w_i_dot = w_i
    #         if w_i < 0:
    #             w_i_dot *= self.n / (np.linalg.norm(np.matmul(inv_sqrt_C, y[i])) ** 2)
    #         w_dot[i] = w_i_dot
    #
    #     self.print('w_dot {}', w_dot)
    #
    #     self.C = (1 + self.c_1 * del_h_sigma - self.c_1 - self.c_mu * self.sum_w) * self.C + \
    #               self.c_1 * np.matmul(self.p_c, self.p_c.transpose()) + \
    #               self.c_mu * sum([w_dot[i] * np.matmul(y[i], y[i].transpose()) for i, y_i in enumerate(y)])
    #     self.print('_C {}', self.C)
    #
    #     # x = np.empty((self.mean.size, len(evaluations)))
    #     # for i, e in enumerate(evaluations):
    #     #     x[:, i] = e[1]
    #     #     y[:, i]
    #
    #     #     samples = np.empty((self.mean.size, len(evaluations)))
    #     #
    #     # for i, e in enumerate(evaluations):
    #     #     samples[:, i] = e[1]
    #     #
    #     # self.mean = np.mean(samples, 1)
    #     # self.covariance = np.cov(samples, ddof=1) * self.variance_scales
    #
    #     self.p_sigma = np.real(self.p_sigma)
    #     self.sigma = np.real(self.sigma)
    #     self.p_c = np.real(self.p_c)
    #     self.m = np.real(self.m)
    #     self.C = np.real(self.C)
    #
    #     self.recorder.accumulate(evaluations, self.mean(), self.variance(), self.C, self.sigma, self.p_c,
    #                               self.p_sigma)
