from tools.optimization.optimizer import Bernoulli


class CenteredBernoulli(Bernoulli):
    
    def __init__(self, p: float = .5, scale: float = 1.0):
        super().__init__(p, a=-scale, b=scale)
