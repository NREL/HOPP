"""
Holds class that defines a grid model for a plant with locally 
distributed generation (e.g. a wind farm or hybrid generation plant).

M. Sinner 6/11/19
"""
import numpy as np
import json
# import cable

class PlantGridModel():
    """
    Model the admittance matrix of a hybrid generation plant.

    Assumes a balanced (symmetric) 3-phase system.

    Attributes:
        admittance_matrix - (N+1)x(N+1) array - admittance matrix for 
                                                the plant grid.
        nodes - (N+1)x3 array - X,Y,Z coordinates for each node in the
                                plant.
        node_labels - (N+1) list of str - label for each node in plant. 
        N - int - number of nodes, EXCLUDING the slack bus.
        nominal_properties - dict - nominal properties for complex:
                                        - power [VA]
                                        - voltage [V]
                                        - current [A]
                                        - admittance [S]
        admittance_matrix_pu - (N+1)x(N+1) array - admittance matrix in 
                                                   the `per unit' 
                                                   (scaled) units of 
                                                   measurement.
    """

    def __init__(self,
                 grid_connect_coordinates, 
                 grid_connect_label='Grid connection'):
        """
        Instantiate FarmGridModel object.

        Inputs:
            grid_connect_coordinates - 1x3 array - X,Y,Z coordinates of 
                                                   the node that
                                                   connects the plant to
                                                   the grid at large. 
            grid_connect_label - str - Label of the grid connection 
                                       node. 
                                       Default: 'Grid connection'.

        Outputs:
            (self) - PlantGridModel object     
        """
        
        # Initialize the admittance matrix
        self.admittance_matrix = np.array([[0+0j]])

        # Produce error if GridConnectCoordinates are list etc
        if type(grid_connect_coordinates) is list or \
           (type(grid_connect_coordinates is np.ndarray) and \
            np.shape(grid_connect_coordinates) != (1,3)):
            TypeError('Please specify GridConnectCoordinates as 1x3 (2D)' +
                      'numpy array of [[X, Y, Z]].')
        else:
            self.nodes = grid_connect_coordinates
        
        if type(grid_connect_label) is str:
            self.node_labels = [grid_connect_label]
        elif type(grid_connect_label) is list:
            self.node_labels = grid_connect_label
        
        # Update number of nodes in system
        self.N = np.shape(self.nodes)[0] - 1

    def add_nodes(self, new_node_coordinates, new_node_labels):
        """
        Add node(s) to the list.

        Inputs:
            new_node_coordinates - nx3 array - X,Y,Z coordinates of n
                                               new nodes being added to
                                               the plant grid.
            new_node_labels - list - labels of n new nodes being added 
                                     to the plant grid.
        
        Outputs:
            none (updates self)
        """
        self.nodes = np.append(self.nodes, new_node_coordinates, axis=0)
        if type(new_node_labels) is str:
            self.node_labels = self.node_labels + [new_node_labels]
        elif type(new_node_labels) is list:
            self.node_labels = self.node_labels + new_node_labels

        # Update number of nodes in system
        self.N = np.shape(self.nodes)[0] - 1
        
        # Pad admittance matrix with zeros to represent no connections 
        # with new nodes.
        self.admittance_matrix = np.pad(self.admittance_matrix,
                                        (0, self.N + 1 - 
                                            np.shape(self.admittance_matrix)[0]
                                            ),
                                        mode='constant',
                                        constant_values=0)

    def add_connections(self,
                        node_pairs, 
                        cable,
                        length_method='square',
                        manual_length=None,
                        grid_frequency=60,
                        ): 
        """
        Add cables to model.
        
        Lines are weighted edges of the (undirected) graph. Edge weight 
        denotes the complex admittance of the line.

        Calculates various quantities and then calls add_edge.
        Optionally, add_edge can be called directly.
        
        Inputs:
            node_pairs - 2D list - 2 node numbers OR 2 node labels per 
                                   outer list entry
            cable - Cable - instantiated cable object specifying the 
                            cable properties (per meter)
            length_method - str - method for determining line length
                                  ('square', 'direct', 'manual')
            manual_length - float - manual entry length of line [m]
                                    (only used if 
                                    length_method='manual')
            line_type - str - underground cables or overhead 
                              transmission lines
                              ('underground', 'overhead')
            phase_separation_distance - float - distance separating 
                                                phase lines [m]
            resistivity - float - electrical resistivity of line 
                                  material [ohm-m]
            grid_frequency - float - nominal grid frequency [Hz]
            relative_permittivity - float - relative permittivity of 
                                            cable insulation [-]
            relative_permeability - float - relative permeability (of
                                            cable insulation?) [-]
        Outputs:
            none (updates self)
        """
        
        resistance_per_meter = cable.resistance

        inductive_reactance_per_meter = 2*np.pi*grid_frequency*cable.inductance

        capacitive_shunt_reactance_per_meter = 1/\
                (2*np.pi*grid_frequency*cable.shunt_capacitance)

        # Apply line properties to the various lines and add to edges
        for i_node_pair in range(len(node_pairs)):
            n1_coord, n1_index = self.extract_node_coordinates(node_pairs[
                                                               i_node_pair][0])
            n2_coord, n2_index = self.extract_node_coordinates(node_pairs[
                                                               i_node_pair][1])
        
            length = self.line_length(n1_coord, n2_coord, length_method, 
                                      manual_length)

            series_resistance = resistance_per_meter * length
            series_reactance = inductive_reactance_per_meter * length
            series_impedance = series_resistance + 1j*series_reactance
            
            shunt_impedance = 1j*capacitive_shunt_reactance_per_meter*length/2

            self.add_edge(n1_index, n2_index, -1/series_impedance)
            self.add_edge(n2_index, n1_index, -1/series_impedance)
            self.update_diagonal_element(n1_index,
                                         1/series_impedance+1/shunt_impedance)
            self.update_diagonal_element(n2_index,
                                         1/series_impedance+1/shunt_impedance)

    def extract_node_coordinates(self, node_index_label):
        """
        Get coordinates for a given node.

        Inputs:
            node_index_label - str OR int - either the label or index of
                                            the desired node
        
        Outputs:
            (coordinates) - 1x3 array - X,Y,Z coordinates of node.
        """
        if type(node_index_label) is str:
            node_index_label = self.node_labels.index(node_index_label)
        
        return self.nodes[node_index_label], node_index_label

    def line_length(self, 
                    node_1_coord, 
                    node_2_coord, 
                    length_method, 
                    manual_length=None):
        """
        Calculate the length of a transmission line between two nodes.
        
        Inputs:
            node_1_coord - 1x3 array - X,Y,Z coordinates of node at one
                                       end of the line. 
            node_2_coord - 1x3 array - X,Y,Z coordinates of node at 
                                       other end of the line. 
            length_method - str - method for determining line length
                                  ('square', 'direct', 'manual')
            manual_length - float - manual entry length of line [m]
                                    (only used if 
                                    length_method='manual')
        
        Outputs:
            (length) - float - length of the transmission line.
        """
        if length_method is 'square':
            return np.sum(abs(node_1_coord - node_2_coord))
        elif length_method is 'direct':
            return np.linalg.norm(node_1_coord - node_2_coord)
        elif length_method is 'manual':
            if manual_length is None:
                ValueError('\'manual\' length method requires a scalar ' + \
                           'length to be provided in manual_length.')
            return manual_length
        else:
            ValueError('lengthMethod specified is invalid. Please use' +
            '\'square\', \'direct\', or \'manual\'.') 
    
    def single_phase_power(self, node_voltages):
        """
        Use admittance matrix to calculate powers, given voltages.
        
        Inputs:
            node_voltages - (N+1)x1 array - complex nodal voltages
        
        Outputs:
            (node_phase_powers) - (N+1)x1 array - complex nodal powers
                                                  on each phase
        """
        return np.diag(node_voltages.T) @ \
               np.conj(self.admittance_matrix) @ \
               np.conj(node_voltages)
    
    def three_phase_power(self, node_voltages):
        """
        Calculate the 3-phase power, given voltages.

        Inputs:
            node_voltages - (N+1)x1 array - complex nodal voltages
        
        Outputs:
            (node_powers) - (N+1)x1 array - complex nodel powers (over
                                            all three phases).
        """
        return 3*self.single_phase_power(node_voltages)

    def add_edge(self, node1, node2, edge_weight):
        """
        Add an edge to the farm grid admittance matrix.

        Inputs:
            node1 - int - index of first node in edge pair
            node2 - int - index of second node in edge pair
            edge_weight - complex - weight of connecting edge 

        Outputs:
            none (updates self)
        """
        self.admittance_matrix[node1, node2] = edge_weight

    def update_diagonal_element(self, node, weight):
       """
       Update admittance matrix so that it remains valid.

       Inputs:
           node - int - index of node whose diagonal entry is being 
                        updated
           weight - complex - added weight to the diagonal entry

        Outputs:
           none (updates self)
       """
       self.admittance_matrix[node, node] = \
           self.admittance_matrix[node, node] + weight
    
    def assign_nominal_quantities(self, nominal_power, nominal_voltage):
        """
        Set and calculate nominal quantities for system

        Inputs:
            nominal_power - float - nominal real power of the system
            nominal_voltage - float - nominal real voltage of the system
        
        Outputs:
            none (updates self)
        """

        # Assign passed-in values
        self.nominal_properties = {
            'power' : nominal_power,
            'voltage' : nominal_voltage
        }
        
        # Find other nominal values
        self.nominal_properties['current'] = nominal_power/\
                                             (np.sqrt(3)*nominal_voltage)
        nominal_impedance = nominal_voltage/ (np.sqrt(3) * \
                                 self.nominal_properties['current'])
        self.nominal_properties['admittance'] = 1/nominal_impedance
    
    @property
    def admittance_matrix_pu(self):
        """
        Convert admittance matrix to `per unit' units
        """
        # Check nominal properties have been assigned
        if not hasattr(self, 'nominal_properties'):
            raise ValueError('Nominal properties must be set before per unit' \
                             'admittance matrix can be calculated.')
        

        # Convert admittance matrix
        return self.admittance_matrix/self.nominal_properties['admittance']

class FlorisGrid(PlantGridModel):
    """
    Wrapper for PlantGridModel that takes in a FLORIS object.
    """

    def __init__(self, input_file, grid_connect_coordinates=None, labels=None, 
                 construction_method='minimal length', nominal_voltage=14e3):
        """
        Instantiate FlorisGrid object (which inherits from 
        PlantGridModel)

        Inputs:
            input_file - string - path to FLORIS json input file. 
            grid_connect_coordinates - 1x3 array - X,Y,Z coordinates of 
                                                   the node that
                                                   connects the plant to
                                                   the grid at large. If
                                                   not provided, will 
                                                   place near first 
                                                   turbine location.
            labels - list of str - Labels of the grid connection node 
                                   turbine nodes.
            construction_method - str - method for building the farm 
                                        grid. Either:
                                           row-wise
                                           minimal length
            nominal_voltage - float - nominal voltage of the farm grid.

        Outputs:
            (self) - FlorisGrid object 
        """
        
        # Read in data from json
        with open(input_file) as file: 
            json_data = json.load(file)
        
        json_farm = json_data['farm']['properties']
        json_turb = json_data['turbine']['properties']
        
        N = len(json_farm['layout_x'])
        x_coords = np.array(json_farm['layout_x'], ndmin=2).T
        y_coords = np.array(json_farm['layout_y'], ndmin=2).T
        z_coords = np.zeros((N,1))
        coordinates = np.concatenate((x_coords, y_coords, z_coords), axis=1)

        # Create grid connection, if none specified.
        if not grid_connect_coordinates:
            grid_connect_coordinates = np.array([[min(x_coords)[0]-100,
                                                  min(y_coords)[0]-100,
                                                  0]])
        # Add nodes to model
        if labels == None:
            grid_connect_label = 'G'
            turbine_labels = ['T'+str(t) for t in range(1,N+1)]
        elif len(labels) == N:
            grid_connect_label = 'G'
            turbine_labels = labels
        else:
            grid_connect_label = labels[0]
            turbine_labels = labels[1:]

        PlantGridModel.__init__(self, grid_connect_coordinates, 
                                      grid_connect_label)    
        PlantGridModel.add_nodes(self, coordinates, turbine_labels)


        # Calculate maximum power using Cp curve
        nominal_power = json_turb['power_thrust_table']['power'][-1] * \
                        0.5*json_farm['air_density'] * \
                        (np.pi*(json_turb['rotor_diameter']/2)**2) * \
                        json_turb['power_thrust_table']['wind_speed'][-1]**3* \
                        json_turb['generator_efficiency'] 
        # TODO: Why is this more than 5MW for example_input.json?
        PlantGridModel.assign_nominal_quantities(self, nominal_power, 
                                                 nominal_voltage)
        
        # Add cables in default configuration
        cable1 = cable.CableByGeometry(700/1000**2)
        # TODO: what is a good nominal cable size?

        # Build grid according to a pre-selected method.
        # row-wise
        # minimum connection
        # user specified connections
        if construction_method == 'row-wise':
            self.construct_row_wise(cable1)
        elif construction_method == 'minimal length':
            self.construct_Prims(cable1)
        #TODO: elif construction_method == 'user specified':
            #self.construct_user_specified() # This one will require inputs
        else:
            NameError('Construction method chosen is not defined')        


    def construct_row_wise(self, cable):
        """
        Connect plant grid using by joining horizontal rows in farm.

        Inputs:
            cable - Cable - instantiated cable object specifying the 
                            cable properties (per meter)
        
        Outputs:
            none (updates self)
        """
            
        x_coords = self.nodes[1:,0]
        y_coords = self.nodes[1:,1]
        row_leaders = []

        # Connect each row
        for row_y in set(y_coords):
            turbines_in_row = [self.node_labels[t+1] for t in range(self.N) \
                                                     if y_coords[t] == row_y]
            row_x_coordinates = x_coords[y_coords == row_y].tolist()
            sorted_turbines = [turbines_in_row[i] 
                               for i in np.argsort(row_x_coordinates)]
            node_pairs = [[sorted_turbines[t], sorted_turbines[t+1]] \
                          for t in range(len(sorted_turbines)-1)]
            PlantGridModel.add_connections(self, node_pairs, cable)
            row_leaders = row_leaders + [sorted_turbines[0]]
        
        # Connect row leaders
        leader_y_coordinates = [self.extract_node_coordinates(T)[0][1] 
                                for T in row_leaders]
        sorted_leaders = [row_leaders[i] 
                          for i in np.argsort(leader_y_coordinates)]
        node_pairs = [[sorted_leaders[t], sorted_leaders[t+1]] \
                      for t in range(len(sorted_leaders)-1)]
        PlantGridModel.add_connections(self, node_pairs, cable)
        
        # Connect to grid
        leader_coordinates = [self.extract_node_coordinates(T)[0] 
                              for T in row_leaders] 
        leader_distances = [np.linalg.norm(leader_coordinates[t]-self.nodes[0])
                            for t in range(len(row_leaders))]
        connect_to_grid = row_leaders[np.argmin(leader_distances)]
        grid_connect = self.node_labels[0]
        PlantGridModel.add_connections(self, [[grid_connect, connect_to_grid]],
                                       cable)

        # end of construct_row_wise

    def construct_Prims(self, cable):
        """
        Connect plant grid using (greedily) `minimal' spanning tree 
        according to Prim's algorithm. 
        https://en.wikipedia.org/wiki/Prim%27s_algorithm

        Inputs:
            cable - Cable - instantiated cable object specifying the 
                            cable properties (per meter)
        
        Outputs:
            none (updates self)
        """
        
        # Find all possible edge lengths (assume same cable for each)
        A_Eucl = np.zeros([self.N+1, self.N+1])
        for i in range(self.N+1):
            for j in range(self.N+1):
                A_Eucl[i,j] = np.linalg.norm(self.nodes[i]-self.nodes[j])

        # Establish tree (initialize with grid connection)
        V_T = [0]
        V_not_T = list(np.arange(1, self.N+1))
        E_T = []
        
        while V_not_T:
            new_edge_options = A_Eucl[np.ix_(V_T,V_not_T)]#.reshape([len(V_T), len(V_not_T)])
            new_edge = np.unravel_index(new_edge_options.argmin(), 
                                        new_edge_options.shape)
            E_T.append([V_T[new_edge[0]], V_not_T[new_edge[1]]])
            V_T.append(V_not_T[new_edge[1]])
            V_not_T.remove(V_not_T[new_edge[1]])
        
        PlantGridModel.add_connections(self, E_T, cable)

        # end of construct_Prims