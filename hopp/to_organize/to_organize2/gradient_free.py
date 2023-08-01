import numpy as np
import time
from math import log

class GeneticAlgorithm():
    """a simple genetic algorithm"""

    def __init__(self):

        # inputs
        self.bits = np.array([]) # array of ints same length as design_variables. 
        self.bounds = np.array([]) # array of tuples same length as design_variables
        self.variable_type = np.array([]) # array of strings same length as design_variables ('int' or 'float')
        self.objective_function = None # takes design_variables as an input and outputs the objective values (needs to account for any constraints already)
        self.max_generation = 100
        self.population_size = 0
        self.crossover_rate = 0.1
        self.mutation_rate = 0.01
        self.tol = 1E-6
        self.convergence_iters = 5
        
        # internal variables, you could output some of this info if you wanted
        self.design_variables = np.array([]) # the desgin variables as they are passed into self.objective function
        self.nbits = 0 # the total number of bits in each chromosome
        self.nvars = 0 # the total number of design variables
        self.parent_population = np.array([]) # 2D array containing all of the parent individuals
        self.offspring_population = np.array([]) # 2D array containing all of the offspring individuals
        self.parent_fitness = np.array([]) # array containing all of the parent fitnesses
        self.offspring_fitness = np.array([]) # array containing all of the offspring fitnesses
        self.discretized_variables = {} # a dict of arrays containing all of the discretized design variable

        # outputs
        self.solution_history = np.array([])
        self.optimized_function_value = 0.0
        self.optimized_design_variables = np.array([])


    def initialize_population(self):

        self.parent_population = np.random.randint(0,high=2,size=(self.population_size,self.nbits))
        self.offspring_population = np.zeros_like(self.parent_population)

    
    def initialize_limited(self):
        """initialize the population with only a limited number of ones. Use this if having a full random initialization
        would violate constraints most of the time"""

        n_ones = 1

        self.parent_population = np.zeros((self.population_size,self.nbits),dtype=int)
        for i in range(self.population_size):
            self.parent_population[i][0:n_ones] = 1
            np.random.shuffle(self.parent_population[i])
            # self.parent_population[i][-11:-1] = np.random.randint(0,high=2,size=10)
        
        self.offspring_population = np.zeros_like(self.parent_population)


    def chromosome_2_variables(self,chromosome):  
        """convert the binary chromosomes to design variable values"""      

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

    
    def optimize_ga(self,initialize="random",crossover="random",print_progress=True,save_progress=False,start_individual=np.array([])):
        """run the genetic algorithm"""

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
        print("initialize population")
        if self.population_size%2 == 1:
            self.population_size += 1

        if initialize == "random":
            self.initialize_population()
        elif initialize == "limit":
            self.initialize_limited()

        if len(start_individual) > 0:
            self.parent_population[0][:] = start_individual[:]

        # initialize the fitness arrays
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

        print("start optimization")
        while converged==False and ngens < self.max_generation:
            # crossover
            if crossover=="random":
                self.crossover()
            elif crossover=="chunk":
                self.chunk_crossover()
            elif crossover=="matrix":
                # I haven't tested this very much
                self.matrix_crossover()
            elif crossover=="both":
                # I haven't tested this very much
                self.crossover()
                self.chunk_crossover()

            # mutation
            self.mutate()

            # determine fitness of offspring
            # print("")
            for i in range(self.population_size):
                # if print_progress == True:
                    # print ("\033[A                             \033[A")
                    # print(i)
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

            # save the intermediate progress of the optimization
            if save_progress:
                if ngens%save_progress == 0:
                    file = open('progress.txt', 'w')
                    file.write('Best solution: %s'%(np.min(self.parent_fitness)) + '\n')
                    self.chromosome_2_variables(self.parent_population[np.argmin(self.parent_fitness)])
                    file.write('Design Variables: %s'%(self.design_variables) + '\n')
                    file.close()

        # Assign final outputs
        self.solution_history = self.solution_history[0:ngens]
        self.optimized_function_value = np.min(self.parent_fitness)
        self.chromosome_2_variables(self.parent_population[np.argmin(self.parent_fitness)])
        self.optimized_design_variables = self.design_variables


    def crossover(self):
        # Random crossover

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

    
    def chunk_crossover(self):
        # Organized crossover (maintain portions of the chromosome in order)
        # set offspring equal to parents
        self.offspring_population[:,:] = self.parent_population[:,:]

        # mate conscutive pairs of parents (0,1),(2,3), ...
        # The population is shuffled so this does not need to be randomized
        for i in range(int(self.population_size/2)):
            # trade bits in the offspring
            crossover_loc = int(np.random.rand(1)*self.nbits)
            begin1 = self.offspring_population[2*i][0:crossover_loc]
            end1 = self.offspring_population[2*i][crossover_loc:self.nbits]
            begin2 = self.offspring_population[2*i+1][0:crossover_loc]
            end2 = self.offspring_population[2*i+1][crossover_loc:self.nbits]
            self.offspring_population[2*i] = np.append(begin1,end2)
            self.offspring_population[2*i+1] = np.append(begin2,end1)

    
    def matrix_crossover(self):
        # Haven't tested this very much
        # organize the matrix
        N = int(np.sqrt(len(self.parent_population[0])))
        M1 = np.zeros((N,N))
        M2 = np.zeros((N,N))
        C1 = np.zeros((N,N))
        C2 = np.zeros((N,N))

        for i in range(int(self.population_size/2)):

            rc = np.random.randint(0,high=2)
            crossover_loc = np.random.randint(0,high=N)

            for j in range(N):
                M1[j,:] = self.parent_population[2*i][j*N:(j+1)*N]
                M2[j,:] = self.parent_population[2*i+1][j*N:(j+1)*N]

            if rc == 0:
                C1[0:crossover_loc,:] = M2[0:crossover_loc,:]
                C1[crossover_loc:N,:] = M1[crossover_loc:N,:]
                C2[0:crossover_loc,:] = M1[0:crossover_loc,:]
                C2[crossover_loc:N,:] = M2[crossover_loc:N,:]

            elif rc == 1:
                C1[:,0:crossover_loc] = M2[:,0:crossover_loc]
                C1[:,crossover_loc:N] = M1[:,crossover_loc:N]
                C2[:,0:crossover_loc] = M1[:,0:crossover_loc]
                C2[:,crossover_loc:N] = M2[:,crossover_loc:N]
            
            for j in range(N):
                self.offspring_population[2*i][j*N:(j+1)*N] = C1[j,:]
                self.offspring_population[2*i+1][j*N:(j+1)*N] = C2[j,:]


    def mutate(self):
        # Randomly mutate bits of each chromosome
        for i in range(int(self.population_size)):
            # mutate bits in the offspring
            mutate_arr = np.random.rand(self.nbits)
            for j in range(self.nbits):
                if mutate_arr[j] < self.mutation_rate:
                    self.offspring_population[i][j] = (self.offspring_population[i][j]+1)%2




class GreedyAlgorithm():

    # A couple of discrete optimization algorithms with greedy principles
    def __init__(self):

        # inputs
        self.bits = np.array([]) # array of ints same length as design_variables. 0 signifies an integer design variable
        self.bounds = np.array([]) # array of tuples same length as design_variables
        self.variable_type = np.array([]) # array of strings same length as design_variables ('int' or 'float')
        self.objective_function = None # takes design_variables as an input and outputs the objective values (needs to account for any constraints already)
        
        # internal variables, you could output some of this info if you wanted
        self.design_variables = np.array([])
        self.nbits = 0
        self.nvars = 0
        self.parent_population = np.array([])
        self.offspring_population = np.array([])
        self.parent_fitness = 0.0
        self.offspring_fitness = 0.0
        self.discretized_variables = {} 

        # outputs
        self.solution_history = np.array([])
        self.optimized_function_value = 0.0
        self.optimized_design_variables = np.array([])


    def chromosome_2_variables(self,chromosome):        
        """convert the binary chromosomes to design variable values"""
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


    def optimize_greedy(self,initialize="ones"):
        # A simple greedy algorithm. Evaluate the objective after switching each bit
        # one at a time. Keep the switched bit that results in the best improvement.
        # Stop after there is no improvement.

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
                int_bits = int(np.ceil(log(int_range,2)+1))
                self.bits[i] = int_bits
            self.nbits += self.bits[i]
        

        # initialize the fitness
        self.parent_fitness = 0.0
        self.offspring_fitness = 0.0

        # initialize the population
        if initialize == "ones":
            done = False
            while done == False:
                self.parent_population = np.zeros(self.nbits,dtype=int)
                self.parent_population[0] = 1
                np.random.shuffle(self.parent_population)
                # self.parent_population[-11:-1] = np.random.randint(0,high=2,size=10)
                # self.parent_population[-6] = 1
                # self.parent_population[-1] = 1
                self.chromosome_2_variables(self.parent_population)
                self.parent_fitness = self.objective_function(self.design_variables)
                print(self.parent_fitness)
                if self.parent_fitness < 1000.0:
                    done = True
        elif initialize=="zeros":
            self.parent_population = np.zeros(self.nbits,dtype=int)
            # self.parent_population[-6] = 1
            # self.parent_population[-1] = 1
        elif initialize=="random":
            done = False
            while done == False:
                self.parent_population = np.random.randint(0,high=2,size=self.nbits)
                self.chromosome_2_variables(self.parent_population)
                self.parent_fitness = self.objective_function(self.design_variables)
                if self.parent_fitness < 1000.0:
                    done = True

        # initialize the offspring population
        self.offspring_population = np.zeros_like(self.parent_population)
        self.offspring_population[:] = self.parent_population[:]

        # initialize the parent fitness
        self.chromosome_2_variables(self.parent_population)
        self.parent_fitness = self.objective_function(self.design_variables)
        
        # initialize optimization
        self.solution_history = np.array([self.parent_fitness])
        converged = False
        best_population = np.zeros(self.nbits)

        while converged==False:
            # loop through every bit
            best_fitness = self.parent_fitness
            best_population[:] = self.parent_population[:]
            for i in range(self.nbits):
                self.offspring_population[:] = self.parent_population[:]
                self.offspring_population[i] = (self.parent_population[i]+1)%2

                # check the fitness
                self.chromosome_2_variables(self.offspring_population)
                self.offspring_fitness = self.objective_function(self.design_variables)

                # check the performance, see if it is the best so far
                if self.offspring_fitness < best_fitness:
                    best_fitness = self.offspring_fitness
                    best_population[:] = self.offspring_population[:]
            
            # check convergence
            if best_fitness == self.parent_fitness:
                converged = True
            
            # update values if not converged
            else:
                print(best_fitness)
                self.solution_history = np.append(self.solution_history,best_fitness)
                self.parent_population[:] = best_population[:]
                self.parent_fitness = best_fitness

        # final outputs
        self.optimized_function_value = self.parent_fitness
        self.chromosome_2_variables(self.parent_population)
        self.optimized_design_variables = self.design_variables


    def optimize_switch(self,initialize="random",print_progress=True):

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
                int_bits = int(np.ceil(log(int_range,2)+1))
                self.bits[i] = int_bits
            self.nbits += self.bits[i]
        
        init = True
        while init == True:
        # initialize the population
            if initialize == "limit":
                nones = 1
                self.parent_population = np.zeros(self.nbits,dtype=int)
                self.parent_population[0:nones] = 1
                np.random.shuffle(self.parent_population)
            else:
                self.parent_population = np.random.randint(0,high=2,size=self.nbits)
            self.offspring_population = np.zeros_like(self.parent_population)
            self.offspring_population[:] = self.parent_population[:]

            # initialize the fitness
            self.parent_fitness = 0.0
            self.offspring_fitness = 0.0

            self.chromosome_2_variables(self.parent_population)
            self.parent_fitness = self.objective_function(self.design_variables)
            if self.parent_fitness != 1E6:
                init = False
        
        # initialize the optimization
        converged = False
        converged_counter = 0
        self.solution_history = np.array([self.parent_fitness])
        index = 1
        random_method = 0

        # initialize the order array (determines the order of sweeping through the variables)
        order = np.arange(self.nbits)

        last_solution = self.parent_fitness
       
        while converged==False:
            # check if we've gone through every bit
            ind = index%self.nbits
            if ind == 0:
                # check if there has been any change since the last phase
                if last_solution == self.parent_fitness:
                    converged_counter += 1
                else:
                    last_solution = self.parent_fitness
                    converged_counter = 0

                # check convergence
                if converged_counter >= 3:
                    converged = True

                # shuffle the order array and change the phase
                np.random.shuffle(order)
                random_method = (random_method+1)%3
                if print_progress == True:
                    if random_method == 0:
                        print("explore")
                    if random_method == 1:
                        print("switch row")
                    if random_method == 2:
                        print("switch col")

            # set offpring equal to parent

            self.offspring_population[:] = self.parent_population[:]

            # this is the explore phase. Switch a bit, evaluate, and see if we should keep it
            if random_method == 0:
                # switch the value of the appropriate index
                self.offspring_population[order[ind]] = (self.parent_population[order[ind]]+1)%2

                # check the fitness
                self.chromosome_2_variables(self.offspring_population)
                self.offspring_fitness = self.objective_function(self.design_variables)

                # check if we should keep the proposed change
                if self.offspring_fitness < self.parent_fitness:
                    self.solution_history = np.append(self.solution_history,self.offspring_fitness)
                    self.parent_fitness = self.offspring_fitness
                    self.parent_population[:] = self.offspring_population[:]
                    if print_progress == True:
                        print(self.offspring_fitness)

            # this is the first switch phase, switch adjacent bits (only makes sense if they are arranged spatially in a matrix)
            elif random_method == 1:
                # organize the matrix
                N = int(np.sqrt(len(self.parent_population)))
                M = np.zeros((N,N))
                for i in range(N):
                    M[i,:] = self.offspring_population[i*N:(i+1)*N]

                row = order[ind]%N
                col = int(order[ind]/N)

                # switch adjacent numbers
                t1 = M[row][col]
                t2 = M[(row+1)%N][col]

                if t1 != t2:
                    M[row][col] = t2
                    M[(row+1)%N][col] = t1
                    
                    for i in range(N):
                        self.offspring_population[i*N:(i+1)*N] = M[i][:]
                    # check the fitness
                    self.chromosome_2_variables(self.offspring_population)
                    self.offspring_fitness = self.objective_function(self.design_variables)

                    if self.offspring_fitness < self.parent_fitness:
                        self.solution_history = np.append(self.solution_history,self.offspring_fitness)
                        self.parent_fitness = self.offspring_fitness
                        self.parent_population[:] = self.offspring_population[:]
                        if print_progress == True:
                            print(self.offspring_fitness)

            # this is the second switch phase, switch adjacent bits in the other dimension (only makes sense if they are arranged spatially in a matrix)
            elif random_method == 2:
                # organize the matrix
                N = int(np.sqrt(len(self.parent_population)))
                M = np.zeros((N,N))
                for i in range(N):
                    M[i][:] = self.offspring_population[i*N:(i+1)*N]

                row = order[ind]%N
                col = int(order[ind]/N)

                # switch adjacent numbers
                t1 = M[row][col]
                t2 = M[row][(col+1)%N]

                if t1 != t2:
                    M[row][col] = t2
                    M[row][(col+1)%N] = t1
                    
                    for i in range(N):
                        self.offspring_population[i*N:(i+1)*N] = M[i][:]
                    # check the fitness
                    self.chromosome_2_variables(self.offspring_population)
                    self.offspring_fitness = self.objective_function(self.design_variables)

                    if self.offspring_fitness < self.parent_fitness:
                        self.solution_history = np.append(self.solution_history,self.offspring_fitness)
                        self.parent_fitness = self.offspring_fitness
                        self.parent_population[:] = self.offspring_population[:]
                        if print_progress == True:
                            print(self.offspring_fitness)

            # increment the counter
            index += 1

        # final output values
        self.optimized_function_value = self.solution_history[-1]
        self.chromosome_2_variables(self.parent_population)
        self.optimized_design_variables = self.design_variables



if __name__=="__main__":

    def simple_obj(x):
        return x[0]+x[1]

    def rosenbrock_obj(x):
        return (1-x[0])**2 + 100.0*(x[1]-x[0]**2)**2

    def ackley_obj(x):
        p1 = -20.0*np.exp(-0.2*np.sqrt(0.5*(x[0]**2+x[1]**2)))
        p2 = np.exp(0.5*(np.cos(2.*np.pi*x[0]) + np.cos(2.0*np.pi*x[1]))) + np.e + 20.0
        return p1-p2

    def rastrigin_obj(x):
        A = 10.0
        n = len(x)
        tot = 0
        for i in range(n):
            tot += x[i]**2 - A*np.cos(2.0*np.pi*x[i])
        return A*n + tot


    import matplotlib.pyplot as plt

    # from mpl_toolkits.mplot3d import Axes3D
    # X = np.arange(-5, 5, 0.02)
    # Y = np.arange(-5, 5, 0.02)
    # X, Y = np.meshgrid(X, Y)
    # Z = np.zeros_like(X)
    # for i in range(np.shape(Z)[0]):
    #     for j in range(np.shape(Z)[1]):
    #         Z[i][j] = rastrigin_obj(np.array([X[i][j],Y[i][j]]))
    
    # # Plot the surface.
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, Z,linewidth=0, antialiased=False)

    # plt.show()

    ga = GeneticAlgorithm()
    ga.bits = np.array([8,8])
    ga.bounds = np.array([(0.0,1.),(0.,1.2)])
    ga.variable_type = np.array(["int","int"])
    ga.population_size = 50
    ga.max_generation = 100
    ga.objective_function = rastrigin_obj
    ga.crossover_rate = 0.1
    ga.mutation_rate = 0.01
    ga.convergence_iters = 25
    ga.tol = 1E-8

    ga.optimize_ga()
    print("optimal function value: ", ga.optimized_function_value)
    print("optimal design variables: ", ga.optimized_design_variables)
    print("nbits: ", ga.nbits)
    plt.plot(ga.solution_history)
    plt.show()
    
