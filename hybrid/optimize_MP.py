from hybrid.scenario import Scenario
from hybrid.optimizer_base import Optimizer
from parameters.parameter_utils import *

from itertools import product
import pprint

def subspace_samples(boundaries, n_intervals_per_axis, tolerance, n_dim):
    interval_width = 1.0 / n_intervals_per_axis
    divided_axis = np.arange(boundaries[0] + interval_width / 2, boundaries[1] - interval_width / 2 + tolerance,
                             interval_width)
    full_axis = np.linspace(interval_width / 2, 1 - interval_width / 2, n_intervals_per_axis)

    arrays = [divided_axis]
    for i in range(n_dim - 1):
        arrays.append(full_axis)
    return list(product(*arrays))


def run_optimizer(boundaries, n_intervals_per_axis, tolerance, n_dim, id):
    """
    Runs an instance of optimization on a subset of the sample space
    :param boundaries: min and max of vector values that can be sampled by this optimizer
    :param n_intervals_per_axis: n_processes * n_intervals_per_process_per_axis
    :param tolerance: some tolerance for dividing axes
    :param n_dim: number of dimensions
    :param id: id of process
    :return: tuple: (max_output, max_coords, n_iteration, output_vec)
    """
    subsamples = subspace_samples(boundaries, n_intervals_per_axis, tolerance, n_dim)
    # n_samples = len(subsamples)
    # idx_interest = -1
    #
    # def gridded_sampling(_, n):
    #     s = subsamples[n]
    #     return s
    #
    # def eval_irr(vec):
    #     return vec[idx_interest]
    #
    # def stop_after_all_samples(n, max, max2):
    #     if n >= n_samples:
    #         return True
    #     else:
    #         return False
    #
    # def save_outputs(n, sample, val):
    #     all_outputs[id * n_samples + n] = val
    #
    # opt = Optimizer(defaults, input_data, output_info, run_systems)
    # idx_interest = [i for i, x in enumerate(opt.scenario.outputs_names) if x == 'capacity_factor'][-1] # generic system capacity factor
    # opt.setup(sampling_func=gridded_sampling, output_eval_func=eval_irr, stopping_cond=stop_after_all_samples,
    #           output_store_func=save_outputs)
    # return opt.optimize()


def init(all_outputs_shared):
    global all_outputs
    # global optimal
    all_outputs = all_outputs_shared.get_obj()
    # optimal = optimal_shared.get_obj()


def process_outputs(plot_config_by_index, scenario, n_processes, n_intervals_per_axis, n_dim, tol, optimal, all_outputs):

    print("Highest weighted value:\n\t", optimal[0])
    print("Coordinates of optimal:\n\t", optimal[1])
    print("Parameters of optimal:\n\t")


    pprint.pprint(scenario.input_data_from_coordinates(optimal[1]))
    print("\nOutput values of optimal:")
    print_output_vals(scenario.output_values(optimal[3]))

    return scenario.output_values(optimal[3])


    """
    all_samples = []
    for i in range(n_processes):
        bounds = (i / n_processes, (i + 1) / n_processes)
        samples = subspace_samples(bounds, n_intervals_per_axis, tol, scenario.vec_length)
        all_samples += samples

    #########################################
    # Plot settings
    #########################################

    # configure 2d slice by marking desired x-axis variable as -1 and y-axis as -2
    # all other variables set to which interval to set as constant for that variable
    title = "Hybrid IRR [%]"

    if len(all_samples) >= 3:

        if len(plot_config_by_index) > 1:
            plot_2d_slice(plot_config_by_index, all_samples, n_intervals_per_axis, all_outputs, scenario, title)
        else:
            fig = plot.figure()
            ax = fig.add_subplot(111)
            Y = [i for i in all_outputs]
            ax.plot([r[0] for r in all_samples], Y)
            # ax.plot_surface(X, Y, results)
            ax.set_title(title)
            plot.figtext(0.5, 0.01, scenario.outputs_map[0], wrap=True, horizontalalignment='center', fontsize=8)
    """