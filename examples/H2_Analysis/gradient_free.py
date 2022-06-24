import numpy as np
import time
from math import log

class GeneticAlgorithm():
    """
    A simple genetic algorithm, which performs uniform crossover and elitism.
    
    INPUT VARIABLES
    bits: an array of ints same length as design_variables that defines the number of divisions considered (2**n)
    bounds: an array of tuples same length as design_variables that describes the bounds of each variable
    variable_type: an array of strings same length as design_variables ('int' or 'float')
    objective_function: a function handle, takes design_variables as an input and outputs the objective values.
                        Needs to account for any constraints already and assumes the objective should be minimized.
    max_generation: int, the maximum number of generations to allow the genetic algorithm to run
    population_size: int, the desired population size
    crossover_rate: float between 0 and 1, the probability of crossover between bits
    mutation_rate: float between 0 and 1, the probability that any individual bit will mutate
    tol: float, the absolute tolerance in determinining convergence 
    convergence_iters: int, the number of generations to determine convergence, in which the best individual does
                       not improve more than tol.

    INTERNAL VARIABLES
    design_variables: array, the desgin variables as they are passed into the objective function
    nbits: int, the total number of bits in each chromosome
    nvars: int, the total number of design variables
    parent_population: 2D array containing all of the parent individuals
    offspring_population: 2D array containing all of the offspring individuals
    parent_fitness: array containing all of the parent fitnesses
    offspring_fitness: array containing all of the offspring fitnesses
    discretized_variables: a dict of arrays containing all of the discretized design variables

    OUTPUTS
    solution_history: array, the best objective function value of each generation
    optimized_function_value: float, the best objective function value
    optimized_design_variables: array, the design variables associated with optimized_function_value
    """

    def __init__(self):

        # inputs
        self.bits = np.array([])
        self.bounds = np.array([])
        self.variable_type = np.array([])
        self.objective_function = None
        self.max_generation = 100
        self.population_size = 0
        self.crossover_rate = 0.1
        self.mutation_rate = 0.01
        self.tol = 1E-6
        self.convergence_iters = 5
        
        # internal variables, you could output some of this info if you wanted
        self.design_variables = np.array([])
        self.nbits = 0
        self.nvars = 0
        self.parent_population = np.array([])
        self.offspring_population = np.array([])
        self.parent_fitness = np.array([])
        self.offspring_fitness = np.array([])
        self.discretized_variables = {}

        # outputs
        self.solution_history = np.array([])
        self.optimized_function_value = 0.0
        self.optimized_design_variables = np.array([])


    def initialize_population(self):
        """
        Initialize the parent and offspring populations. Parents are initialized randomly, offspring
        are initialized to zeros. 
        """
        self.parent_population = np.random.randint(0, high=2, size=(self.population_size, self.nbits))
        self.offspring_population = np.zeros_like(self.parent_population)


    def chromosome_2_variables(self,chromosome):  
        """
        Convert the binary chromosomes to design variable values.

        chormosome: array, the binary array representing an individual within the population.
        """      

        first_bit = 0
        float_ind = 0

        for i in range(self.nvars):
            binary_value = 0
            for j in range(self.bits[i]):
                binary_value += chromosome[first_bit+j]*2**j
            first_bit += self.bits[i]

            if self.variable_type[i] == "float":
                self.design_variables[i] = self.discretized_variables["float_var%s"%float_ind][binary_value]
                float_ind += 1

            elif self.variable_type[i] == "int":
                self.design_variables[i] = self.bounds[i][0] + binary_value

    
    def optimize_ga(self,print_progress=False):
        """
        Run the genetic algorithm.
        """

        if print_progress:
            print("start GA")
        # determine the number of design variables and initialize
        self.nvars = len(self.variable_type)
        self.design_variables = np.zeros(self.nvars)
        float_ind = 0
        for i in range(self.nvars):
            if self.variable_type[i] == "float":
                ndiscretizations = 2**self.bits[i]
                self.discretized_variables["float_var%s"%float_ind] = np.linspace(self.bounds[i][0],self.bounds[i][1],ndiscretizations)
                float_ind += 1

        # determine the total number of bits
        for i in range(self.nvars):
            if self.variable_type[i] == "int":
                int_range = self.bounds[i][1] - self.bounds[i][0]
                int_bits = int(np.ceil(log(int_range,2)))
                self.bits[i] = int_bits
            self.nbits += self.bits[i]        

        # initialize the population
        if print_progress:
            print("initialize population")
        if self.population_size%2 == 1:
            self.population_size += 1

        self.initialize_population()

        # initialize the fitness arrays
        if print_progress:
            print("initialize fitness")
        self.parent_fitness = np.zeros(self.population_size)
        self.offspring_fitness = np.zeros(self.population_size)

        # initialize fitness of the parent population
        for i in range(self.population_size):
            self.chromosome_2_variables(self.parent_population[i])
            self.parent_fitness[i] = self.objective_function(self.design_variables)

        converged = False
        ngens = 1
        generation = 1
        difference = self.tol * 10000.0
        self.solution_history = np.zeros(self.max_generation+1)
        self.solution_history[0] = np.min(self.parent_fitness)

        if print_progress:
            print("start optimization")
        while converged==False and ngens < self.max_generation:
            self.crossover()
            self.mutate()

            for i in range(self.population_size):
                self.chromosome_2_variables(self.offspring_population[i])
                self.offspring_fitness[i] = self.objective_function(self.design_variables)

            # rank the total population from best to worst
            total_fitness = np.append(self.parent_fitness,self.offspring_fitness)
            ranked_fitness = np.argsort(total_fitness)[0:int(self.population_size)]

            # take the best. Might switch to some sort of tournament, need to read more about what is better
            # for now I've decided to only keep the best members of the population. I have a large population in 
            # the problems I've run with this so I assume sufficient diversity in the population is maintained from that
            total_population = np.vstack([self.parent_population,self.offspring_population])
            self.parent_population[:,:] = total_population[ranked_fitness,:]
            self.parent_fitness[:] = total_fitness[ranked_fitness]
            
            # store solution history and wrap up generation
            self.solution_history[generation] = np.min(self.parent_fitness)

            if generation > self.convergence_iters:
                difference = self.solution_history[generation-self.convergence_iters] - self.solution_history[generation]
            else:
                difference = 1000
            if abs(difference) <= self.tol:
                converged = True
            
            # shuffle up the order of the population
            shuffle_order = np.arange(1,self.population_size)
            np.random.shuffle(shuffle_order)
            shuffle_order = np.append([0],shuffle_order)
            self.parent_population = self.parent_population[shuffle_order]
            self.parent_fitness = self.parent_fitness[shuffle_order]
            if print_progress==True:
                print(self.parent_fitness[0])

            generation += 1
            ngens += 1

        # Assign final outputs
        self.solution_history = self.solution_history[0:ngens]
        self.optimized_function_value = np.min(self.parent_fitness)
        self.chromosome_2_variables(self.parent_population[np.argmin(self.parent_fitness)])
        self.optimized_design_variables = self.design_variables

    def crossover(self):
        """
        Perform uniform crossover between individual bits of each parent
        """

        # set offspring equal to parents
        self.offspring_population[:,:] = self.parent_population[:,:]

        # mate conscutive pairs of parents (0,1),(2,3), ...
        # The population is shuffled so this does not need to be randomized
        for i in range(int(self.population_size/2)):
            # trade bits in the offspring
            crossover_arr = np.random.rand(self.nbits)
            for j in range(self.nbits):
                if crossover_arr[j] < self.crossover_rate:
                    self.offspring_population[2*i][j], self.offspring_population[2*i+1][j] = self.offspring_population[2*i+1][j], self.offspring_population[2*i][j]

    def mutate(self):
        """
        Randomly mutate bits of each chromosome.
        """
        for i in range(int(self.population_size)):
            # mutate bits in the offspring
            mutate_arr = np.random.rand(self.nbits)
            for j in range(self.nbits):
                if mutate_arr[j] < self.mutation_rate:
                    self.offspring_population[i][j] = (self.offspring_population[i][j]+1)%2
