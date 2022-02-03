from pyomo.opt import TerminationCondition


class DispatchProblemState:
    """Class for tracking dispatch problem solve state and metrics"""

    def __init__(self):
        self._start_time = ()
        self._n_days = ()
        self._termination_condition = ()
        self._solve_time = ()
        self._objective = ()
        self._upper_bound = ()
        self._lower_bound = ()
        self._constraints = ()
        self._variables = ()
        self._non_zeros = ()
        self._gap = ()
        self._n_non_optimal_solves = 0

    def store_problem_metrics(self, solver_results, start_time, n_days, objective_value):
        self.start_time = start_time
        self.n_days = n_days
        self.termination_condition = str(solver_results.solver.termination_condition)
        try:
            self.solve_time = solver_results.solver.time
        except AttributeError:
            self.solve_time = solver_results.solver.wallclock_time
        self.objective = objective_value
        self.upper_bound = solver_results.problem.upper_bound
        self.lower_bound = solver_results.problem.lower_bound
        self.constraints = solver_results.problem.number_of_constraints
        self.variables = solver_results.problem.number_of_variables
        self.non_zeros = solver_results.problem.number_of_nonzeros

        # solver_results.solution.Gap not define
        if solver_results.problem.upper_bound != 0.0:
            self.gap = (abs(solver_results.problem.upper_bound - solver_results.problem.lower_bound)
                        / abs(solver_results.problem.upper_bound))
        elif solver_results.problem.lower_bound == 0.0:
            self.gap = 0.0
        else:
            self.gap = float('inf')

        if not solver_results.solver.termination_condition == TerminationCondition.optimal:
            self._n_non_optimal_solves += 1

    def _update_metric(self, metric_name, value):
        data = list(getattr(self, metric_name))
        data.append(value)
        setattr(self, '_' + metric_name, tuple(data))

    @property
    def start_time(self) -> tuple:
        return self._start_time

    @start_time.setter
    def start_time(self, start_hour: int):
        self._update_metric('start_time', start_hour)

    @property
    def n_days(self) -> tuple:
        return self._n_days

    @n_days.setter
    def n_days(self, solve_days: int):
        self._update_metric('n_days', solve_days)

    @property
    def termination_condition(self) -> tuple:
        return self._termination_condition

    @termination_condition.setter
    def termination_condition(self, condition: str):
        self._update_metric('termination_condition', condition)

    @property
    def solve_time(self) -> tuple:
        return self._solve_time

    @solve_time.setter
    def solve_time(self, time: float):
        self._update_metric('solve_time', time)

    @property
    def objective(self) -> tuple:
        return self._objective

    @objective.setter
    def objective(self, objective_value: float):
        self._update_metric('objective', objective_value)

    @property
    def upper_bound(self) -> tuple:
        return self._upper_bound

    @upper_bound.setter
    def upper_bound(self, bound: float):
        self._update_metric('upper_bound', bound)

    @property
    def lower_bound(self) -> tuple:
        return self._lower_bound

    @lower_bound.setter
    def lower_bound(self, bound: float):
        self._update_metric('lower_bound', bound)

    @property
    def constraints(self) -> tuple:
        return self._constraints

    @constraints.setter
    def constraints(self, constraint_count: int):
        self._update_metric('constraints', constraint_count)

    @property
    def variables(self) -> tuple:
        return self._variables

    @variables.setter
    def variables(self, variable_count: int):
        self._update_metric('variables', variable_count)

    @property
    def non_zeros(self) -> tuple:
        return self._non_zeros

    @non_zeros.setter
    def non_zeros(self, non_zeros_count: int):
        self._update_metric('non_zeros', non_zeros_count)

    @property
    def gap(self) -> tuple:
        return self._gap

    @gap.setter
    def gap(self, mip_gap: int):
        self._update_metric('gap', mip_gap)

    @property
    def n_non_optimal_solves(self) -> int:
        return self._n_non_optimal_solves
