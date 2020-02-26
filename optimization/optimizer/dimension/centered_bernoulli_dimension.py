from optimization.optimizer.dimension.bernoulli_dimension import BernoulliDimension


class CenteredBernoulliDimension(BernoulliDimension):
    
    def __init__(self, p: float = .5, scale: float = 1.0):
        super().__init__(p, a=-scale, b=scale)
