import pprint


class Candidate:
    """
    Simulation inputs to be optimized
    """
    
    def __init__(
            self
            ) -> None:
        pass
    
    def __repr__(self) -> str:
        return pprint.pformat(vars(self))
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __setitem__(self,
                    key: str,
                    value: object
                    ) -> None:
        self.__setattr__(key, value)
