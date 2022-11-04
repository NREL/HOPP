%%  RODeO batch Prep file
%
%     Creates batch file by quickly assembling different sets of runs and 
%       can create multiple batch files for parallel processing
%
%     Steps to add new fields
%       1. Add entry into Batch_header (Second section)
%       2. Add value(s) for new fields (Third section)


%% Prepare data to populate batch file.
clear all, close all, clc
Project_name = 'Central_vs_distributed'; 
% Project_name = 'Example';

dir1 = 'C:\Users\jeichman\Documents\gamsdir\projdir\RODeO\';   % Set directory to send files
dir2 = [dir1,'Projects\',Project_name,'\Batch_files\'];
cd(dir1); 

% Define overall properties
GAMS_loc = 'C:\GAMS\win64\24.8\gams.exe';
GAMS_file= {'Storage_dispatch_v22_1'};      % Define below for each utility (3 file options)
GAMS_lic = 'license=C:\GAMS\win64\24.8\gamslice.txt';
files_to_create = 2;  % Select the number of batch files to create

outdir = ['Projects\',Project_name,'\Output'];
indir  = ['Projects\',Project_name,'\Data_files\TXT_files'];

% Load filenames
files_tariff = dir([dir1,indir]);
files_tariff2={files_tariff.name}';  % Identify files in a folder    
for i0=1:length(files_tariff2) % Remove items from list that do not fit criteria
    if ((~isempty(strfind(files_tariff2{i0},'additional_parameters'))+...
         ~isempty(strfind(files_tariff2{i0},'renewable_profiles'))+...
         ~isempty(strfind(files_tariff2{i0},'controller_input_values'))+...
         ~isempty(strfind(files_tariff2{i0},'building')))>0)     % Skip files called "additional_parameters" or "renewable profile" 
    else
        load_file1(i0)=~isempty(strfind(files_tariff2{i0},'.txt'));       % Find only txt files
    end
end 
files_tariff2=files_tariff2(load_file1);    clear load_file1 files_tariff

Batch_header = struct;
Batch_header.elec_rate_instance = {};
Batch_header.H2_price_hourly = {};
Batch_header.Hydrogen_consumed_hourly = {};
Batch_header.Input_power_baseload_hourly = {};
Batch_header.NG_price_hourly = {};
Batch_header.ren_prof_instance = {};
Batch_header.load_prof_instance = {};
Batch_header.energy_price_inst = {};
Batch_header.AS_price_inst = {};
Batch_header.outdir = {};
Batch_header.indir = {};

Batch_header.gas_price_instance = {};
Batch_header.zone_instance = {};
Batch_header.year_instance = {};

Batch_header.input_cap_instance = {};
Batch_header.output_cap_instance = {};
Batch_header.price_cap_instance = {};

Batch_header.Apply_input_cap_inst = {};
Batch_header.Apply_output_cap_inst = {};
Batch_header.max_output_cap_inst = {};
Batch_header.allow_import_instance = {};

Batch_header.input_LSL_instance = {};
Batch_header.output_LSL_instance = {};
Batch_header.Input_start_cost_inst = {};
Batch_header.Output_start_cost_inst = {};
Batch_header.input_efficiency_inst = {};
Batch_header.output_efficiency_inst = {};

Batch_header.input_cap_cost_inst = {};
Batch_header.output_cap_cost_inst = {};
Batch_header.input_FOM_cost_inst = {};
Batch_header.output_FOM_cost_inst = {};
Batch_header.input_VOM_cost_inst = {};
Batch_header.output_VOM_cost_inst = {};
Batch_header.input_lifetime_inst = {};
Batch_header.output_lifetime_inst = {};
Batch_header.interest_rate_inst = {};

Batch_header.in_heat_rate_instance = {};
Batch_header.out_heat_rate_instance = {};
Batch_header.storage_cap_instance = {};
Batch_header.storage_set_instance = {};
Batch_header.storage_init_instance = {};
Batch_header.storage_final_instance = {};
Batch_header.reg_cost_instance = {};
Batch_header.min_runtime_instance = {};
Batch_header.ramp_penalty_instance = {};

Batch_header.op_length_instance = {};
Batch_header.op_period_instance = {};
Batch_header.int_length_instance = {};

Batch_header.lookahead_instance = {};
Batch_header.energy_only_instance = {};
Batch_header.file_name_instance = {};
Batch_header.H2_consume_adj_inst = {};
Batch_header.H2_price_instance = {};
Batch_header.H2_use_instance = {};
Batch_header.base_op_instance = {};
Batch_header.NG_price_adj_instance = {};
Batch_header.Renewable_MW_instance = {};

Batch_header.CF_opt_instance = {};
Batch_header.run_retail_instance = {};
Batch_header.one_active_device_inst = {};

Batch_header.current_int_instance = {};
Batch_header.next_int_instance = {};
Batch_header.current_stor_intance = {};
Batch_header.current_max_instance = {};
Batch_header.max_int_instance = {};
Batch_header.read_MPC_file_instance = {};

fields1 = fieldnames(Batch_header);

%% Set values to vary 
relationship_length = zeros(numel(fields1),1);      % Create matrix to track size of variations for each variable
for i0 = 1:numel(fields1)
    if strcmp(fields1{i0},'elec_rate_instance')
        Batch_header.elec_rate_instance.val = strrep(files_tariff2,'.txt','');
    end   
    if strcmp(fields1{i0},'H2_price_hourly')
        Batch_header.H2_price_hourly.val = {'additional_parameters_hourly'};        
    end
    if strcmp(fields1{i0},'Hydrogen_consumed_hourly')
        Batch_header.Hydrogen_consumed_hourly.val = {'H2_consumption_central_hourly','H2_consumption_distributed_hourly'};        
    end
    if strcmp(fields1{i0},'Input_power_baseload_hourly')
        Batch_header.Input_power_baseload_hourly.val = {'Input_power_baseload_hourly'};        
    end
    if strcmp(fields1{i0},'NG_price_hourly')
        Batch_header.NG_price_hourly.val = {'NG_price_Price1_hourly'};        
    end
    if strcmp(fields1{i0},'ren_prof_instance')
        Batch_header.ren_prof_instance.val = {'renewable_profiles_none_hourly'};
    end
    if strcmp(fields1{i0},'load_prof_instance')
        Batch_header.load_prof_instance.val = {'Additional_load_Load1_hourly'};
    end
    if strcmp(fields1{i0},'energy_price_inst')
        Batch_header.energy_price_inst.val = {'Energy_prices_empty_hourly'};
    end
    if strcmp(fields1{i0},'AS_price_inst')
        Batch_header.AS_price_inst.val = {'Ancillary_services_hourly'};
    end
    if strcmp(fields1{i0},'outdir')
        [status,msg] = mkdir(outdir);       % Create output file if it doesn't exist yet  
        Batch_header.outdir.val = {outdir}; % Reference is dynamic from location of batch file (i.e., exclue 'RODeO\' in the filename for batch runs but include for runs within GAMS GUI)
    end
    if strcmp(fields1{i0},'indir')
        Batch_header.indir.val = {indir};       % Reference is dynamic from location of batch file (i.e., exclue 'RODeO\' in the filename for batch runs but include for runs within GAMS GUI)
    end
    
    if strcmp(fields1{i0},'gas_price_instance')
        Batch_header.gas_price_instance.val = {'NA'};
    end
    if strcmp(fields1{i0},'zone_instance')
        Batch_header.zone_instance.val = {'NA'};
    end
    if strcmp(fields1{i0},'year_instance')
        Batch_header.year_instance.val = {'NA'};
    end
    
    if strcmp(fields1{i0},'input_cap_instance')
        Batch_header.input_cap_instance.val = {'1000'};
    end
    if strcmp(fields1{i0},'output_cap_instance')
        Batch_header.output_cap_instance.val = {'0'};
    end
    if strcmp(fields1{i0},'price_cap_instance')
        Batch_header.price_cap_instance.val = {'10000'};
    end
    
    if strcmp(fields1{i0},'Apply_input_cap_inst')
        Batch_header.Apply_input_cap_inst.val = {'0'};
    end
    if strcmp(fields1{i0},'Apply_output_cap_inst')
        Batch_header.Apply_output_cap_inst.val = {'0'};
    end
    if strcmp(fields1{i0},'max_output_cap_inst')
        Batch_header.max_output_cap_inst.val = {'inf'};
    end
    if strcmp(fields1{i0},'allow_import_instance')
        Batch_header.allow_import_instance.val = {'1'};
    end
    
    if strcmp(fields1{i0},'input_LSL_instance')
        Batch_header.input_LSL_instance.val = {'0'};
    end
    if strcmp(fields1{i0},'output_LSL_instance')
        Batch_header.output_LSL_instance.val = {'0'};
    end
    if strcmp(fields1{i0},'Input_start_cost_inst')
        Batch_header.Input_start_cost_inst.val = {'0'};
    end
    if strcmp(fields1{i0},'Output_start_cost_inst')
        Batch_header.Output_start_cost_inst.val = {'0'};
    end
    if strcmp(fields1{i0},'input_efficiency_inst')
        Batch_header.input_efficiency_inst.val = {'0.613668913'};
    end
    if strcmp(fields1{i0},'output_efficiency_inst')
        Batch_header.output_efficiency_inst.val = {'1'};
    end
    
    if strcmp(fields1{i0},'input_cap_cost_inst')
        Batch_header.input_cap_cost_inst.val = {'0'};
    end
    if strcmp(fields1{i0},'output_cap_cost_inst')
        Batch_header.output_cap_cost_inst.val = {'0'};
    end
    if strcmp(fields1{i0},'input_FOM_cost_inst')
        Batch_header.input_FOM_cost_inst.val = {'0'};
    end
    if strcmp(fields1{i0},'output_FOM_cost_inst')
        Batch_header.output_FOM_cost_inst.val = {'0'};
    end
    if strcmp(fields1{i0},'input_VOM_cost_inst')
        Batch_header.input_VOM_cost_inst.val = {'0'};
    end
    if strcmp(fields1{i0},'output_VOM_cost_inst')
        Batch_header.output_VOM_cost_inst.val = {'0'};
    end
    if strcmp(fields1{i0},'input_lifetime_inst')
        Batch_header.input_lifetime_inst.val = {'0'};
    end
    if strcmp(fields1{i0},'output_lifetime_inst')
        Batch_header.output_lifetime_inst.val = {'0'};
    end
    if strcmp(fields1{i0},'interest_rate_inst')
        Batch_header.interest_rate_inst.val = {'0'};
    end
    
    if strcmp(fields1{i0},'in_heat_rate_instance')
        Batch_header.in_heat_rate_instance.val = {'0'};
    end
    if strcmp(fields1{i0},'out_heat_rate_instance')
        Batch_header.out_heat_rate_instance.val = {'0'};
    end
    if strcmp(fields1{i0},'storage_cap_instance')
        Batch_header.storage_cap_instance.val = {'6'};
    end
    if strcmp(fields1{i0},'storage_set_instance')
        Batch_header.storage_set_instance.val = {'1'};
    end    
    if strcmp(fields1{i0},'storage_init_instance')
        Batch_header.storage_init_instance.val = {'0.5'};
    end
    if strcmp(fields1{i0},'storage_final_instance')
        Batch_header.storage_final_instance.val = {'0.5'};
    end
    if strcmp(fields1{i0},'reg_cost_instance')
        Batch_header.reg_cost_instance.val = {'0'};
    end
    if strcmp(fields1{i0},'min_runtime_instance')
        Batch_header.min_runtime_instance.val = {'0'};
    end
    if strcmp(fields1{i0},'ramp_penalty_instance')
        Batch_header.ramp_penalty_instance.val = {'0'};
    end
    
    if strcmp(fields1{i0},'op_length_instance')
        Batch_header.op_length_instance.val = {'8760'};
    end
    if strcmp(fields1{i0},'op_period_instance')
        Batch_header.op_period_instance.val = {'8760'};
    end
    if strcmp(fields1{i0},'int_length_instance')
        Batch_header.int_length_instance.val = {'1'};
    end
    
    if strcmp(fields1{i0},'lookahead_instance')
        Batch_header.lookahead_instance.val = {'0'};
    end
    if strcmp(fields1{i0},'energy_only_instance')
        Batch_header.energy_only_instance.val = {'1'};
    end
    if strcmp(fields1{i0},'file_name_instance')
        % 'file_name_instance' created in the next section (default value of 0)
        Batch_header.file_name_instance.val = {'0'};
    end
    if strcmp(fields1{i0},'H2_consume_adj_inst')
        Batch_header.H2_consume_adj_inst.val = {'0.9'};
    end
    if strcmp(fields1{i0},'H2_price_instance')
        Batch_header.H2_price_instance.val = {'6'};
    end
    if strcmp(fields1{i0},'H2_use_instance')
        Batch_header.H2_use_instance.val = {'1'};
    end
    if strcmp(fields1{i0},'base_op_instance')
        Batch_header.base_op_instance.val = {'0','1'};
    end
    if strcmp(fields1{i0},'NG_price_adj_instance')
        Batch_header.NG_price_adj_instance.val = {'1'};
    end
    if strcmp(fields1{i0},'Renewable_MW_instance')
        Batch_header.Renewable_MW_instance.val = {'0'};
    end
    
    if strcmp(fields1{i0},'CF_opt_instance')
        Batch_header.CF_opt_instance.val = {'0'};
    end
    if strcmp(fields1{i0},'run_retail_instance')
        Batch_header.run_retail_instance.val = {'1'};
    end
    if strcmp(fields1{i0},'one_active_device_inst')
        Batch_header.one_active_device_inst.val = {'1'};
    end
    
    if strcmp(fields1{i0},'current_int_instance')
        Batch_header.current_int_instance.val = {'-1'};
    end
    if strcmp(fields1{i0},'next_int_instance')
        Batch_header.next_int_instance.val = {'1'};
    end
    if strcmp(fields1{i0},'current_stor_intance')
        Batch_header.current_stor_intance.val = {'0.5'};
    end
    if strcmp(fields1{i0},'current_max_instance')
        Batch_header.current_max_instance.val = {'0.8'};
    end
    if strcmp(fields1{i0},'max_int_instance')
        Batch_header.max_int_instance.val = {'Inf'};
    end
    if strcmp(fields1{i0},'read_MPC_file_instance')
        Batch_header.read_MPC_file_instance.val = {'0'};
    end
    relationship_length(i0) = length(Batch_header.(fields1{i0}).val);
end

% Create 2D matrix of all possible combinations
relationship_matrix = [];
for i0=1:numel(fields1)
    num_items = numel(Batch_header.(fields1{i0}).val);
    if i0==1,
        relationship_matrix = Batch_header.(fields1{i0}).val;
    else
        [M0,N0] = size(relationship_matrix);               
        relationship_matrix_interim = relationship_matrix;  % Copy matrix to repeat
        fields1_val = Batch_header.(fields1{i0}).val;       % Capture all values for selected field        
        for i1=1:num_items
            add_column = cell(M0,1);                        % Create empty matrix to add to existing matrix
            [add_column{:}] = deal([fields1_val{i1}]);      % Populate matrix with repeated cell item
            relationship_matrix_interim([(i1-1)*M0+1:i1*M0],[1:(N0+1)]) = horzcat(relationship_matrix,add_column); 
        end
        relationship_matrix = relationship_matrix_interim;  % Overwrite matrix with completed 
    end   
end
clear i0 M0 N0

%% Define how values are varied between fields
%  Not defining how a field1 is varied over field2 causes the code to default to making the same values for field1 across every field2 
%  Example: F1 = {'0','1'}, F2 = {'10','100'}. Without defining a specific relationship between F1 and F2, four runs will be created (i.e., [F1=0,F2=10], [F1=0,F2=100], [F1=1,F2=10], [F0=1,F1=100])
%  Definition of relationship should be arranged as e.g., input_cap_instance.base_op_instance. 
%  The first item is the row and the second is the column.
%  Example: input_cap_instance = {'900','1000'}; base_op_instance = {'0','1'};
%  Exapmle: The resulting matrix would be         C1        C2
%                                           R1    [900,0]   [900,1]
%                                           R2    [1000,0]  [1000,1]
%  Example: Create matrix to express the items that you want (1=include, 0 = exclude) 
%  Example: Batch_header.input_cap_instance.base_op_instance = [0 1;1 0]; 
%  Setup to work for multiple relationships per field

% Define relationships between fields 
%   (only define one relationship per field)
%   (Do not define the reciprocal relationship... if -> Batch_header.A.B then do not define Batch_header.B.A)

% % Batch_header.input_cap_instance.base_op_instance = [0 1;1 0];

% Adjust relationship matrix based on definitions above
relationship_toggle = ones(prod(relationship_length),1);            % Matrix to turn on and off specific runs (1=include, 0=exclude) (default, before applying exceptions, is to include all runs)
for i0=1:numel(fields1)
    fields2 = fieldnames(Batch_header.(fields1{i0}));
    [M0,N0] = size(fields2);
    if M0==1, continue
    else
        for i1=2:M0
            find_index1 = i0;                                       % Repeat value for i0
            find_index2 = strfind(fields1,fields2(i1));             % Find string in cell array 
            find_index2 = find(not(cellfun('isempty',find_index2)));% Find string in cell array 
            
            find_val1 = Batch_header.(fields1{find_index1}).val;    % Find values in row
            find_val2 = Batch_header.(fields1{find_index2}).val;    % Find values in column
            
            find_rel1 = Batch_header.(fields1{find_index1}).(fields1{find_index2});  % Find relationship between items
            [M1,N1] = size(find_rel1);
            
            for i2=1:M1
                for i3=1:N1
                    find_rel1_val = find_rel1(i2,i3);
                    if find_rel1_val==0
                        find_val3 = {find_val1{i2},find_val2{i3}};
                        compare_fields1 = horzcat(relationship_matrix(:,find_index1),relationship_matrix(:,find_index2));
                        find_row1 = strfind(relationship_matrix(:,find_index1),find_val1{i2});  % Find rows that match omitted items
                        find_row1 = find(not(cellfun('isempty',find_row1)));                    % Find rows that match ommited items

                        find_row2 = strfind(relationship_matrix(:,find_index2),find_val2{i3});  % Find rows that match omitted items
                        find_row2 = find(not(cellfun('isempty',find_row2)));                    % Find rows that match ommited items
                        
                        remove_items = intersect(find_row1,find_row2);
                        relationship_toggle(remove_items) = 0;
                    end                    
                end
            end            
        end
    end
end
clear i0 i1 i2 i3 M0 N0 M1 N1
relationship_matrix_final = relationship_matrix;
relationship_matrix_final(find(relationship_toggle==0),:)=[];
[M0,N0] = size(relationship_matrix_final);


% Create file names
Index_file_name = strfind(fields1,'file_name_instance');
Index_file_name = find(not(cellfun('isempty',Index_file_name)));

Index_elec_rate = strfind(fields1,'elec_rate_instance');
Index_elec_rate = find(not(cellfun('isempty',Index_elec_rate)));

Index_base = strfind(fields1,'base_op_instance');
Index_base = find(not(cellfun('isempty',Index_base)));

Index_CF = strfind(fields1,'H2_consume_adj_inst');
Index_CF = find(not(cellfun('isempty',Index_CF)));

Index_H2_cons = strfind(fields1,'Hydrogen_consumed_hourly');
Index_H2_cons = find(not(cellfun('isempty',Index_H2_cons)));

for i0=1:M0    
    interim1 = relationship_matrix_final{i0,Index_elec_rate};
    Find_underscore1 = strfind(interim1,'_');
    interim1 = interim1(1:Find_underscore1-1);    
    
    interim2 = relationship_matrix_final{i0,Index_base};
    if strcmp(interim2,'0'),     interim2='Flex';
    elseif strcmp(interim2,'1'), interim2='Base';
    end
    
    interim3 = relationship_matrix_final{i0,Index_CF};
    interim3 = ['CF',num2str(round(str2num(interim3)*100,0))];

    interim4 = relationship_matrix_final{i0,Index_H2_cons};
    if strcmp(interim4,'H2_consumption_central_hourly'),     interim4='Central';
    elseif strcmp(interim4,'H2_consumption_distributed_hourly'), interim4='Distributed';
    end
    
    relationship_matrix_final{i0,Index_file_name} = horzcat(interim1,'_',interim2,'_',interim3,'_',interim4);
end


%% Create batch file names for tariffs
c2=1;   % Initialize batch file number
fileID = fopen([dir2,['RODeO_batch',num2str(c2),'.bat']],'wt');

% Create GAMS run command and write to text file
for i0=1:M0
GAMS_batch_init = ['"',GAMS_loc,'" "',GAMS_file,'" ',GAMS_lic];
    for i1=1:N0
        GAMS_batch{i1,1} = horzcat([' --',fields1{i1},'="',relationship_matrix_final{i0,i1},'"']);        
    end
    fprintf(fileID,'%s\n\n',[[GAMS_batch_init{:}],[GAMS_batch{:}]]);
    if mod(i0,ceil(M0/files_to_create))==0
        if i0==M0
        else
            fclose(fileID);
            c2=c2+1;            
            fileID = fopen([dir2,['RODeO_batch',num2str(c2),'.bat']],'wt');
        end
    end 
    if mod(i0,100)==0
        disp(['File ',num2str(c2),' : ',num2str(i0),' of ',num2str(M0)]);   % Display progress    
    end    
end
fclose(fileID);
disp([num2str(c2),' batch files created for ',num2str(M0),' model runs (~',num2str(ceil(M0/files_to_create)),' each)']);


