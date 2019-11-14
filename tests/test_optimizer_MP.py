import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plot
import sys
import multiprocessing
from itertools import product
import pytest

tolerance = 1e-4

def subspace_samples(boundaries, n_intervals_per_axis, n_dim):
    int_width = 1.0 / n_intervals_per_axis
    divided_axis = np.arange(boundaries[0] + int_width/2, boundaries[1] - int_width/2 + tolerance, int_width)

    full_axis = np.linspace(int_width/2, 1 - int_width/2, n_intervals_per_axis)

    arrays = [divided_axis]
    for i in range(n_dim - 1):
        arrays.append(full_axis)
    return list(product(*arrays))


# test subspace sampling
n_dim = 4
for process in range(1, 4):
    for j in range(process, 4):
        interval = j*process

        all_samples = []
        for i in range(process):
            bounds = (i / process, (i + 1) / process)
            samples = subspace_samples(bounds, interval, n_dim)
            all_samples += samples
        assert (len(all_samples) == interval ** n_dim)
        uniques = np.unique(all_samples, axis=1)
        assert(len(uniques) == len(all_samples))
        print("Passed test 1 for ", process, " process and ", interval, " intervals")


# using gridded sampling to test multiprocessing
n_processes = 2
n_intervals_per_process_per_axis = 2    # should be an even number for clean division of space
n_intervals_per_axis = n_processes * n_intervals_per_process_per_axis
optimal = []


def run_optimizer(boundaries, id):
    """
    Runs an instance of optimization on a subset of the sample space
    :param boundaries: min and max of vector values that can be sampled by this optimizer
    :return: tuple: (max_output, max_coords, n_iteration, output_vec)
    """

    all_outputs = None
    all_samples = subspace_samples(boundaries, n_intervals_per_axis, n_dim)

    print(len(all_samples))
    n_samples = len(all_samples)

    def gridded_sampling(_, n):
        s = all_samples[n]
        return s

    def eval_irr(sample):
        v = 0
        for i in range(len(sample)):
            v += round(sample[i], 2)*100**(len(sample)-i)
        return v

    def stop_after_all_samples(n, max, max2):
        if n >= n_samples:
            return True
        else:
            return False

    def save_outputs(n, sample, val):
        all_outputs[id * n_samples + n] = val

    def optimize(stopping_condition, sampling_function, output_eval_function, output_store_function):
        """
        Basic optimization routine with stopping condition, sample generation & sample evaluation functions
        :return: tuple with max_output, max_coords, n_iteration
        """
        n_iterations = 0
        max_output = sys.float_info.min
        max_output_prev = max_output
        max_coords = []
        while not stopping_condition(n_iterations, max_output, max_output_prev):
            sample = sampling_function(n_dim, n_iterations)
            output_val = output_eval_function(sample)
            if output_store_function:
                output_store_function(n_iterations, sample, output_val)
            if output_val > max_output:
                max_output_prev = max_output
                max_output = output_val
                max_coords = sample
            n_iterations += 1
        return max_output, max_coords, n_iterations

    return optimize(stop_after_all_samples, gridded_sampling, eval_irr, save_outputs)


def optimal_result(result):
    global optimal
    if not optimal:
        optimal = result
    else:
        if result[0] > optimal[0]:
            optimal = result

@pytest.mark.skip(reason="There is no optimal result being found")
def test_optimizer_MP():

    print("\n\nOptimizing:\n=========================\n")
    # Optimize the technologies over the range of selected inputs to vary within parameter_data

    all_outputs = multiprocessing.Array('d', n_intervals_per_axis ** n_dim )

    pool = multiprocessing.Pool()
    for i in range(n_processes):
        bounds = (i/n_processes, (i+1)/n_processes)
        results = pool.apply_async(run_optimizer, args=(bounds, i), callback=optimal_result)
    pool.close()
    pool.join()

    print("Highest weighted value:\n\t", optimal[0])
    print("Coordinates of optimal:\n\t", optimal[1])
    print("all outputs", all_outputs[:])

    # make a plot
    # inputs to plot as x and y axis. mark x-axis as -1 and y-axis as -2. otherwise which tick value to set a constant
    plot_config_by_index = [-1, 0, -2, 0]

    ticks = []
    coords = []
    cnt = 0
    x_ind = plot_config_by_index.index(-1)
    y_ind = plot_config_by_index.index(-2)
    constant = [i for i in range(len(plot_config_by_index)) if plot_config_by_index[i] == 0]
    results = np.zeros((n_intervals_per_axis, n_intervals_per_axis))
    all_samples = []
    for i in range(n_processes):
        bounds = (i / n_processes, (i + 1) / n_processes)
        samples = subspace_samples(bounds, n_intervals_per_axis, n_dim)
        all_samples += samples
    ticks = np.unique(all_samples).tolist()
    for sample in all_samples:
        keep = True
        for ind in constant:
            if sample[ind] != ticks[plot_config_by_index[ind]]:
                keep = False
        if keep:
            x_pos = ticks.index(sample[x_ind])
            y_pos = ticks.index(sample[y_ind])
            results[x_pos][y_pos] = all_outputs[cnt]
            print("keeping", sample, x_pos, y_pos, all_outputs[cnt], cnt)
        cnt += 1

    print(results)

    X, Y = np.meshgrid(ticks, ticks)
    print(X)
    print(Y)

    fig = plot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, results)
    plot.show()

if __name__ == '__main__':
    test_optimizer_MP()