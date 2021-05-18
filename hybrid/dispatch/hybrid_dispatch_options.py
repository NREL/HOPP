from hybrid.dispatch import (SimpleBatteryDispatchHeuristic,
                             SimpleBatteryDispatch,
                             NonConvexLinearVoltageBatteryDispatch,
                             ConvexLinearVoltageBatteryDispatch)


class HybridDispatchOptions:
    """

    """
    def __init__(self, dispatch_options: dict = None):
        self.battery_dispatch: str = 'simple'
        self.include_lifecycle_count: bool = True
        self.n_look_ahead_periods: int = 48
        self.n_roll_periods: int = 24
        self.log_name: str = 'hybrid_dispatch_optimization.log'
        self.is_test: bool = False

        if dispatch_options is not None:
            for key, value in dispatch_options.items():
                if hasattr(self, key):
                    if type(getattr(self, key)) == type(value):
                        setattr(self, key, value)
                    else:
                        raise ValueError("'{}' is the wrong data type.".format(key))
                else:
                    raise NameError("'{}' is not an attribute in {}".format(key, type(self).__name__))

        self._battery_dispatch_model_options = {
            'heuristic': SimpleBatteryDispatchHeuristic,
            'simple': SimpleBatteryDispatch,
            'non_convex_LV': NonConvexLinearVoltageBatteryDispatch,
            'convex_LV': ConvexLinearVoltageBatteryDispatch}
        if self.battery_dispatch in self._battery_dispatch_model_options:
            self.battery_dispatch_class = self._battery_dispatch_model_options[self.battery_dispatch]
        else:
            raise ValueError("'{}' is not currently a battery dispatch class.".format(self.battery_dispatch))
