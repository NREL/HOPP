%% Updated on 04/11: Remove confusing comments. That is removing texts including Feeder A, B,C, etc. Also, modify some comments to make the file clearer.


clear;

%% Read load data
load Feeder1_P;     % Active powers correpsonding to Feeder 1's buses  
load Feeder1_Q;     % Reactive powers correpsonding to Feeder 1's buses
load Feeder2_P;     % Active powers correpsonding to Feeder 2's buses
load Feeder2_Q;     % Reactive powers correpsonding to Feeder 2's buses
load Feeder3_P;     % Active powers correpsonding to Feeder 3's buses
load Feeder3_Q;     % Reactive powers correpsonding to Feeder 3's buses
load Feeder4_P;     % Active powers correpsonding to Feeder 4's buses
load Feeder4_Q;     % Reactive powers correpsonding to Feeder 4's buses
load Feeder6_P;     % Active powers correpsonding to Feeder 6's buses  
load Feeder6_Q;     % Reactive powers correpsonding to Feeder 6's buses
load Feeder7_P;     % Active powers correpsonding to Feeder 7's buses
load Feeder7_Q;     % Reactive powers correpsonding to Feeder 7's buses
load Feeder8_P;     % Active powers correpsonding to Feeder 8's buses
load Feeder8_Q;     % Reactive powers correpsonding to Feeder 8's buses

%% Build the Matlab-OpenDSS COM interface
DSSObj=actxserver('OpenDSSEngine.DSS');            % Register the COM server (initialization)
if ~DSSObj.Start(0)                                % Start the OpenDSS, and if the registration is unsuccessful, stop the program and remind the user
    disp('Unable to start OpenDSS Engine');
    return
end
DSSText = DSSObj.Text;                             % Define a text interface variable
DSSCircuit = DSSObj.ActiveCircuit;                 % Define a circuit interface variable
DSSText.Command='Compile "C:\Users\fbu\Desktop\Project\AMU\East substation\Master.dss"';   % Specify the directory of OpenDSS master file 

%% Define variables to collect the power flow results
bus_voltages_rect = [];                   % Bus voltage in rectangular coordinate
bus_voltage_magni_pu = [];                % Bus voltage magnitude in per unit.
line_currents = {};                       % Line current
line_powers = {};                         % Line power
elem_names = [];                          % Element names
elem_losses = [];                         % Element loss
total_power = [];                         % System total power                
i_notconverged = 0;                       % Define a variable to record the number of unconverged snapshot power flow solutions
Tap_position_collect_East_South = [];                % Tap changer position
Tap_position_collect_East_North = [];                % Tap changer position


%% Specify load buses (buses with load)
Feeder1_bus_with_load = [1002, 1004, 1007:1010, 1012:1014, 1016:1024, 1026:1029, 1031:1033, 1035, 1037:1046, 1048:1049, 1051, 1053:1054, 1056, 1058, 1061:1062, ...
                         1064, 1066:1068, 1071:1077, 1079:1080, 1083, 1085, 1087, 1090:1092, 1094, 1096, 1098:1099, 1101, 1103:1104, 1106:1108];                                                                                                                     
Feeder2_bus_with_load = [2002:2003, 2005, 2008:2011, 2014:2018, 2020, 2022:2025, 2028:2032, 2034:2035, 2037, 2040:2043, 2045:2056, 2058:2060];       
Feeder3_bus_with_load = [3002, 3004, 3006:3007, 3009:3014, 3016:3021, 3023:3029, 3031:3039, 3041:3045, 3047:3052, 3054, 3056:3067, 3070:3074, 3077:3078, 3081, ...
                         3083:3091, 3093:3099, 3101:3106, 3108:3112, 3114:3117, 3120:3132, 3134:3138, 3141:3155, 3157:3162];                         
Feeder4_bus_with_load = [4004:4008, 4010:4014, 4016, 4019:4022, 4024, 4026, 4028, 4030, 4032:4035, 4037:4040, 4042:4048];  
Feeder6_bus_with_load = 6003:6017;                            
Feeder7_bus_with_load = [7003:7005, 7008:7010, 7012];         
Feeder8_bus_with_load = [8003, 8005:8006, 8008:8010, 8012:8013, 8015:8018, 8020:8023, 8028:8031];        

%% Solve quasi-static time-series power flow via Matlab-OpenDSS interface and collect results
n = length(Feeder1_P(:, 1));                        % Number of hours in one year, i.e., 8760
for i = 1:n
  tic
      %% For each load of Feeder 1, set kW and kVar 
      for k = 1:length(Feeder1_bus_with_load)           % From the 1st bus with load to the last bus with load
          bus_num = Feeder1_bus_with_load(1,k);         % Bus No.
          DSSText.command=[[char('load.Load_'), num2str(bus_num), char('.kW=')]  num2str(Feeder1_P(i, bus_num-1000)) ' kvar='  num2str(Feeder1_Q(i, bus_num-1000)) ''];  % Build bus name and set corresponding kW and kVar
                                                                                                                                                   %  bus_num-1000 specifies the column number that the power corresponds to 
      end    
    
    
      %% For each load of Feeder 2, set kW and kVar 
      for k = 1:length(Feeder2_bus_with_load)          % From the 1st bus with load to the last bus with load
          bus_num = Feeder2_bus_with_load(1,k);        % Bus No.
          DSSText.command=[[char('load.Load_'), num2str(bus_num), char('.kW=')]  num2str(Feeder2_P(i, bus_num-2000)) ' kvar='  num2str(Feeder2_Q(i, bus_num-2000)) ''];   % Build bus name and set corresponding kW and kVar
                                                                                                                                                   %  bus_num-2000 specifies the column number that the power corresponds to
      end     
    
    
      %% For each load of Feeder 3, set kW and kVar 
      for k = 1:length(Feeder3_bus_with_load)          % From the 1st bus with load to the last bus with load
          bus_num = Feeder3_bus_with_load(1,k);        % Bus No.
          DSSText.command=[[char('load.Load_'), num2str(bus_num), char('.kW=')]  num2str(Feeder3_P(i, bus_num-3000)) ' kvar='  num2str(Feeder3_Q(i, bus_num-3000)) ''];   % Build bus name and set corresponding kW and kVar
                                                                                                                                                   %  bus_num-3000 specifies the column number that the power corresponds to
      end
      
      %% For each load of Feeder 4, set kW and kVar 
      for k = 1:length(Feeder4_bus_with_load)          % From the 1st bus with load to the last bus with load
          bus_num = Feeder4_bus_with_load(1,k);        % Bus No.
          DSSText.command=[[char('load.Load_'), num2str(bus_num), char('.kW=')]  num2str(Feeder4_P(i, bus_num-4000)) ' kvar='  num2str(Feeder4_Q(i, bus_num-4000)) ''];   % Build bus name and set corresponding kW and kVar
                                                                                                                                                   %  bus_num-3000 specifies the column number that the power corresponds to
      end

      
      %% For each load of Feeder 6, set kW and kVar 
      for k = 1:length(Feeder6_bus_with_load)           % From the 1st bus with load to the last bus with load
          bus_num = Feeder6_bus_with_load(1,k);         % Bus No.
          DSSText.command=[[char('load.Load_'), num2str(bus_num), char('.kW=')]  num2str(Feeder6_P(i, bus_num-6000)) ' kvar='  num2str(Feeder6_Q(i, bus_num-6000)) ''];  % Build bus name and set corresponding kW and kVar
                                                                                                                                                   %  bus_num-6000 specifies the column number that the power corresponds to 
      end    
    
    
      %% For each load of Feeder 7, set kW and kVar 
      for k = 1:length(Feeder7_bus_with_load)          % From the 1st bus with load to the last bus with load
          bus_num = Feeder7_bus_with_load(1,k);        % Bus No.
          DSSText.command=[[char('load.Load_'), num2str(bus_num), char('.kW=')]  num2str(Feeder7_P(i, bus_num-7000)) ' kvar='  num2str(Feeder7_Q(i, bus_num-7000)) ''];   % Build bus name and set corresponding kW and kVar
                                                                                                                                                   %  bus_num-7000 specifies the column number that the power corresponds to
      end     
    
    
      %% For each load of Feeder 8, set kW and kVar 
      for k = 1:length(Feeder8_bus_with_load)          % From the 1st bus with load to the last bus with load
          bus_num = Feeder8_bus_with_load(1,k);        % Bus No.
          DSSText.command=[[char('load.Load_'), num2str(bus_num), char('.kW=')]  num2str(Feeder8_P(i, bus_num-8000)) ' kvar='  num2str(Feeder8_Q(i, bus_num-8000)) ''];   % Build bus name and set corresponding kW and kVar
                                                                                                                                                   %  bus_num-3000 specifies the column number that the power corresponds to
      end

      %% Solve snapshot power flow
      DSSText.Command='solve';


      %% Convergence checking and record the number of uncoverged snapshot power flow
      DSSSolution = DSSCircuit.Solution;              % Obtain solution information from the interface
      if ~DSSSolution.Converged                       % Check whether the power flow computation converges
          fprintf('The Circuit did not converge. \n');
          i_notconverged = i_notconverged + 1;        % calculate the total number of unconverged power flow computations
          continue;
      end
      
      
      %% Collect power flow results 
      % Collect bus names 
      bus_names = DSSCircuit.AllNodenames;   
      
      % Collect all bus voltages in rectangular coordinate
      bus_voltage_temp = DSSCircuit.AllBusVolts;                                                  % Obtain rectangular bus voltage in each snapshot power flow solution            
      bus_voltages_rect = [bus_voltages_rect; bus_voltage_temp];                                  % Collect rectangular bus voltages in all snapshot power flow solutions 

      % Collect all bus voltage magnitude in p.u.
      bus_voltage_magni_pu_temp = DSSCircuit.AllBusVmagPu;                                        % Obtain bus voltage in p.u. in each snapshot power flow solution 
      bus_voltage_magni_pu = [bus_voltage_magni_pu; bus_voltage_magni_pu_temp];                   % Collect bus voltages in p.u. in all snapshot power flow solutions
      
      % Collect element names and losses
      elem_names = DSSCircuit.AllElementNames;                                                    % Obtain element names
      elem_loss_temp = DSSCircuit.AllElementLosses;                                               % Obtain element losses in each snapshot power flow solution
      elem_losses = [elem_losses; elem_loss_temp];                                                % Collect element losses in all snapshot power flow solutions

      % Collect total power of the entire system
      total_power_temp = DSSCircuit.TotalPower;                                                   % Obtain total power of the entire system in each snapshot power flow solution
      total_power = [total_power; total_power_temp];                                              % Collect total power of the entire system in all snapshot power flow solutions

      % Collect currents and powers of all lines
       currents_DSS_Lines = [];                                                                   % Define a variable to collect line currents in each snapshot power flow solution
       powers_DSS_Lines = [];                                                                     % Define a variable to collect line powers in each snapshot power flow solution
       DSSLines = DSSObj.ActiveCircuit.Lines;                                                     % Specify that the currently activated objects are lines
       DSSActiveCktElement = DSSObj.ActiveCircuit.ActiveCktElement;                               % Returns an interface to the active circuit element (lines).

       
       line_names = {};                              % Names of lines
       line_I_points_nums = [];                      % For each line, define a variable to collect the number of current (or power) variables in rectangular coordinate, e.g., a single phase line has 4 current variables,... 
                                                     % ... i.e., real and image parts of the current which flows into the head of the line, real and image parts of the current which flows out the end of the line 
       i_Line = DSSLines.First;                      % Initializing line NO. as the first line
       while i_Line > 0                              % From the 1st line to the last line
            currents_DSS_Lines = [currents_DSS_Lines, DSSActiveCktElement.Currents];                % Collect line currents in each snapshot power flow solution
            powers_DSS_Lines= [powers_DSS_Lines, DSSActiveCktElement.Powers];                       % Collect line powers in each snapshot power flow solution
            line_names{i_Line, 1} = DSSActiveCktElement.NAME;                                       % Collect line names in each snapshot power flow solution
            line_I_points_nums = [line_I_points_nums; length(DSSActiveCktElement.Currents)];        % Collect the total number of variables correponsing to each line in each snapshot power flow solution
            i_Line = DSSLines.Next;                                                                 % Move to next line in each snapshot power flow solution
       end

       
       line_curr_temp = {};       
       line_power_temp = {};
       num_lines = length(line_names);
       for j = 1:num_lines                                                          % find starting and ending indices of currents for each line 
           indx_str = 1 + sum(line_I_points_nums(1:j-1));
           indx_end = sum(line_I_points_nums(1:j));
           line_curr_temp{1, j} = currents_DSS_Lines(indx_str:indx_end);
           line_power_temp{1, j} = powers_DSS_Lines(indx_str:indx_end);
       end
       
       
      line_currents(i,:) = line_curr_temp;                                          % Collect line currents in all snapshot power flow solutions
      line_powers(i,:) = line_power_temp;                                           % Collect line powers in all snapshot power flow solutions      
       
      
      
      % Collect tap positions
      DSSCircuit.RegControls.Name = 'Reg_contr_A_south';                          % Specify the name of tap changer contoller from which we want to get tap postion                         
      TapChanger1_temp = DSSCircuit.RegControls.TapNumber;                  % Obtain tap position of tap changer (Phase A)
      DSSCircuit.RegControls.Name = 'Reg_contr_B_south';                          % Specify the name of tap changer contoller from which we want to get tap postion
      TapChanger2_temp = DSSCircuit.RegControls.TapNumber;                  % Obtain tap position of tap changer (Phase B)
      DSSCircuit.RegControls.Name = 'Reg_contr_C_south';                          % Specify the name of tap changer contoller from which we want to get tap postion
      TapChanger3_temp = DSSCircuit.RegControls.TapNumber;                  % Obtain tap position of tap changer (Phase C)
      Tap_position_collect_East_South = [Tap_position_collect_East_South; [TapChanger1_temp, TapChanger2_temp, TapChanger3_temp]];   % Collect tap changers positions in all snapshot power flow solutions
      
      DSSCircuit.RegControls.Name = 'Reg_contr_A_north';                          % Specify the name of tap changer contoller from which we want to get tap postion                         
      TapChanger1_temp = DSSCircuit.RegControls.TapNumber;                  % Obtain tap position of tap changer (Phase A)
      DSSCircuit.RegControls.Name = 'Reg_contr_B_north';                          % Specify the name of tap changer contoller from which we want to get tap postion
      TapChanger2_temp = DSSCircuit.RegControls.TapNumber;                  % Obtain tap position of tap changer (Phase B)
      DSSCircuit.RegControls.Name = 'Reg_contr_C_north';                          % Specify the name of tap changer contoller from which we want to get tap postion
      TapChanger3_temp = DSSCircuit.RegControls.TapNumber;                  % Obtain tap position of tap changer (Phase C)
      Tap_position_collect_East_North = [Tap_position_collect_East_North; [TapChanger1_temp, TapChanger2_temp, TapChanger3_temp]];   % Collect tap changers positions in all snapshot power flow solutions
      
toc
end

fprintf('The number of snapshot power flow soultions that do not converge is: %d. \n', i_notconverged);                 % Print the total number of unconverged power flow solutions
