%% Prepare data to populate batch file.
clear all, close all, clc
dir1 = 'C:\Users\jeichman\Documents\gamsdir\projdir\RODeO\';   % Set directory to send files
dir2 = [dir1,'batch_files\'];
cd(dir1); 

% Define overall properties
q = char(39);
GAMS_loc = 'C:\GAMS\win64\24.7\gams.exe';
write_net_meter_files = 'no';
GAMS_file= {'Storage_dispatch_v22_1'};      % Define below for each utility (3 file options)
GAMS_lic = 'license=C:\GAMS\win64\24.7\gamslice.txt';
items_per_batch = 300;  % Select the number of items per batch file created

outdir = 'Output\PGE';
indir = 'Input_files\Default3';

% Load filenames
files_tariff = dir([dir1,'Input_files\Default3\hourly_tariffs\']);
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

batch_header = {'file_name_instance','gas_price_instance','zone_instance','year_instance','input_cap_instance','output_cap_instance',...
                'input_LSL_instance','output_LSL_instance','Input_start_cost_inst','Output_start_cost_inst',...
                'input_efficiency_inst','output_efficiency_inst','in_heat_rate_instance','out_heat_rate_instance',...
                'storage_cap_instance','reg_cost_instance','min_runtime_instance','op_period_instance','int_length_instance',...
                'lookahead_instance','energy_only_instance','H2_consume_adj_inst','H2_price_instance',...
                'H2_use_instance','base_op_instance','NG_price_adj_instance','Renewable_MW_instance',...
                'elec_rate_instance','add_param_instance','ren_prof_instance','current_int_instance',...
                'next_int_instance','current_stor_intance','current_max_instance','max_int_instance','outdir','indir','Load_prof_instance','read_MPC_file_instance'};

%% Create batch file names for regular tariffs
% Variations (expand files created based on following items)
if strcmp(write_net_meter_files,'yes')
     var1 = {'EYnet_Base','EYnet_Flex','SMR','EYnet_FLEXwAS'};  % Operation strategy selected
else var1 = {'EY_Base','EY_Flex','SMR','EY_FLEXwAS'};           % Operation strategy selected
end
var2 = {'1','2','3','4','8','12','24','48','96','168'};        	% Storage Cap Instance (hours at rated capacity)
var3 = {'40CF','60CF','80CF','90CF','95CF'};                    % Capacity factor
var4 = {'0RE','10RE','30RE','50RE','100RE','200RE','300RE','400RE';'0RE','0RE','0RE','0RE','0RE','0RE','0RE','0RE'};  % Installed renewable capacity (% of EY)
var4A= {''};  % Text used to find renewable or DG tariffs
var5 = {'NoRE','PV','WIND'};    % Select renewables to include (none, PV, Wind) (make sure it matches with "renewable_profiles"
renewable_profiles = {'renewable_profiles_none','renewable_profiles_PV','renewable_profiles_WIND'}; % Match var5 and renewable profiles

val1 = [0,0,0];
val2 = [1,2,3,4,8,12,24,48,96,168];
val3 = [0.4,0.6,0.8,0.9,0.95];
val4 = [0,100,300,500,1000,2000,3000,4000;0,0,0,0,0,0,0,0];
[val4R,val4C] = size(val4);
val5 = zeros(1,length(var5));
val6 = [8760,35040,105120;1,0.25,0.0833333333]';    % Col1 = number of intervals, Col2 = interval_length



input_cap_instance_val = [1000,1000,100,1000];
output_cap_instance_val = [0,0,0,0];
input_LSL_instance_val = [0,0.1,0,0.1];
output_LSL_instance_val = [0,0,0,0];
input_efficiency_inst_val = {'0.613668913','0.613668913','58.56278067','0.613668913'};
output_efficiency_inst_val = {'1','1','1','1'};
input_heat_rate_inst_val = {'0','0','274.6045494','0'};
storage_cap_instance_val = val2;
energy_only_instance_val = [1,1,1,0];
H2_consume_adj_inst_val = val3; 
base_op_instance_val = [1,0,1,0];



val4_adj = [1400,2100,2799,3149,3324];  % PV
% val4_adj = [1291,1937,2582,2905,3066];  % WIND

c0=0;   % Initialize
c1=0;   % Initialize batch file counter
c2=1;   % Initialize batch file number
Avoid_items_end = {'_33curt','_40curt'};    %{'_PV','_WIND'};   % Removes the following items
Avoid_items_tariff = {'E20R_','TOU8A_','TOU8R_','DGR_','TOU8CPP_'};   % Removes the following items
Avoid_items_connection = {'_tran_','_pri_'};

%%% Open first batch file
fileID = fopen([dir2,['Batch_Regular_tariffs',num2str(c2),'.bat']],'wt');

for i0=1:length(files_tariff2)
    files_tariff2_short = files_tariff2{i0};
    int1 = find(files_tariff2_short=='_');
    util1 = 1;  % Allows for use of differnt GAMS models in the final batch file
%     if     strcmp(files_tariff2_short(1:int1(1)-1),'PGE'),  util1 = 1;
%     elseif strcmp(files_tariff2_short(1:int1(1)-1),'SCE'),  util1 = 2;
%     elseif strcmp(files_tariff2_short(1:int1(1)-1),'SDGE'), util1 = 3;
%     end
%     if max(strcmp(files_tariff2_short(int1(end):end-4),Avoid_items_end))   % Skip several items
%         continue
%     end
%     if max(strcmp(files_tariff2_short(int1(2):int1(3)),Avoid_items_connection))   % Skip several items
%         continue
%     end    
%     if max(strcmp(files_tariff2_short(int1(1)+1:int1(2)),Avoid_items_tariff))   % Skip several items
%         continue
%     end
    for i1=[1:2]  %1:length(var1)   % Cycle through each variable
    for i2=[5]  %1:length(var2)
    for i3=[4]  %1:length(var3)
    for i4=[1]  %1:length(var4)
    for i5=[1]  %1:length(var5)
        if (strcmp(var1(i1),'SMR') && i4~=1)    % Skip renewable scenarios for SMR
            continue
        end
        if (strfind(var5{i5},'NoRE') && val4(1,i4)~=0) % Skip reneawble scenarios when no renewables are installed
            continue
        end
        var44 = var4;                       % Repeat var4
        val44 = val4;                       % Repeat val4        
        %%% Used to adjust for max PV and max wind
%         var44(2,end) = {[num2str(round(val4_adj(i3)/10,0)),'RE']};
%         val44(2,end) = val4_adj(i3);        % Add line to adjust renewable capacity to max required for each to achieve 100% renpen
        %%%        
        Renewable_MW_instance_val = val44;  % Reset after 33curt, 40curt
% %         if max(strcmp(files_tariff2_short(int1(end)+1:end-4),{'33curt','40curt'}))
% %             if (strcmp(var1(i1),'SMR'))    % Skip all SMR entries for 33curt and 40curt
% %                 continue
% %             end
% %             if i4==1  % Change all values to 100RE for 33curt and 40curt
% %                 var44 = cell(size(var4)); var44(:) = {'100RE'};
% %                 val44 = ones(size(val4))*1000;
% %                 Renewable_MW_instance_val = val44;    
% %             else continue    % Skips the 33curt and 40curt items except for the first
% %             end            
% %         end        
        c0=c0+1;  c1=c1+1;  % Increase for each iteration
        if ~isempty(strfind(files_tariff2_short,var4A{:}))  % Used to select either the first or second line of renewable values (for renewable tariffs)
            if strcmp(var1(i1),'SMR')   % Skip renewable scenarios for SMR
                c0=c0-1; c1=c1-1;       % Remove increased iteration
                continue
            end
            Renewable_MW_instance(c0) = {num2str(Renewable_MW_instance_val(2,i4))};
            file_name_instance(c0) = {[var1{i1},'_',files_tariff2_short(1:end-4),'_',var3{i3},'_',var44{2,i4},'_',var5{i5}]};
        else
%             continue      % Skip all non renewable tariffs
            Renewable_MW_instance(c0) = {num2str(Renewable_MW_instance_val(1,i4))};
            file_name_instance(c0) = {[var1{i1},'_',files_tariff2_short(1:end-4),'_',var3{i3},'_',var44{1,i4},'_',var5{i5}]};
        end
        
        % Determine the resolution of the tariff file
        if     (strfind(files_tariff2_short,'_hourly'))>0, val66 = val6(1,:); additional_parameters='additional_parameters_hourly'; renewable_profiles2=[renewable_profiles{i5},'_hourly']; Load_profile=['basic_building_0','_hourly'];
        elseif (strfind(files_tariff2_short,'_15min'))>0,  val66 = val6(2,:); additional_parameters='additional_parameters_15min';  renewable_profiles2=[renewable_profiles{i5},'_15min'];  Load_profile=['basic_building_0','_15min'];
        elseif (strfind(files_tariff2_short,'_5min'))>0,   val66 = val6(3,:); additional_parameters='additional_parameters_5min';   renewable_profiles2=[renewable_profiles{i5},'_5min'];   Load_profile=['basic_building_0','_5min'];
        else   error('Correct resolution not found');            
        end
        
        gas_price_instance(c0)      = {'NA'};
        zone_instance(c0)           = {'NA'};
        year_instance(c0)           = {'NA'};
        input_cap_instance(c0)      = {num2str(input_cap_instance_val(i1))};
        output_cap_instance(c0)     = {num2str(output_cap_instance_val(i1))};
        input_LSL_instance(c0)      = {num2str(input_LSL_instance_val(i1))};
        output_LSL_instance(c0)     = {num2str(output_LSL_instance_val(i1))};
        Input_start_cost_inst(c0)   = {num2str(0.00001)};
        Output_start_cost_inst(c0)  = {num2str(0)};
        input_efficiency_inst(c0)   = input_efficiency_inst_val(i1);
        output_efficiency_inst(c0)  = output_efficiency_inst_val(i1);
        in_heat_rate_instance(c0)   = input_heat_rate_inst_val(i1);
        out_heat_rate_instance(c0)  = {num2str(0)};
        storage_cap_instance(c0)    = {num2str(storage_cap_instance_val(i2))};
        reg_cost_instance(c0)       = {num2str(0)};
        min_runtime_instance(c0)    = {num2str(0)};
        op_period_instance(c0)      = {num2str(val66(1,1))};
        int_length_instance(c0)     = {num2str(val66(1,2))};
        lookahead_instance(c0)      = {num2str(0)};
        energy_only_instance(c0)    = {num2str(energy_only_instance_val(i1))};
        H2_consume_adj_inst(c0)     = {num2str(H2_consume_adj_inst_val(i3))};
        H2_price_instance(c0)       = {num2str(6)};
        H2_use_instance(c0)         = {num2str(1)};
        base_op_instance(c0)        = {num2str(base_op_instance_val(i1))};
        NG_price_adj_instance(c0)   = {num2str(1)};
        elec_rate_instance(c0)      = {files_tariff2_short(1:end-4)};
        add_param_instance(c0)      = {additional_parameters};
        ren_prof_instance(c0)       = {renewable_profiles2};
        current_int_instance(c0)    = {num2str(-1)};    % Set default parameter
        next_int_instance(c0)       = {num2str(1)};     % Set default parameter
        current_stor_intance(c0)    = {num2str(0)};     % Set default parameter
        current_max_instance(c0)    = {num2str(0)};     % Set default parameter
        max_int_instance(c0)        = {num2str(inf)};   % Set default parameter
        
        % Create GAMS run command and write to text file
        GAMS_batch = ['"',GAMS_loc,'" ',GAMS_file{util1},' ',GAMS_lic,...
                    ' --',batch_header{1},'="',file_name_instance{c0},'"',...
                    ' --',batch_header{2},'="',gas_price_instance{c0},'"',...
                    ' --',batch_header{3},'="',zone_instance{c0},'"',...
                    ' --',batch_header{4},'="',year_instance{c0},'"',...
                    ' --',batch_header{5},'=',input_cap_instance{c0},...
                    ' --',batch_header{6},'=',output_cap_instance{c0},...
                    ' --',batch_header{7},'=',input_LSL_instance{c0},...
                    ' --',batch_header{8},'=',output_LSL_instance{c0},...
                    ' --',batch_header{9},'=',Input_start_cost_inst{c0},...
                    ' --',batch_header{10},'=',Output_start_cost_inst{c0},...
                    ' --',batch_header{11},'=',input_efficiency_inst{c0},...
                    ' --',batch_header{12},'=',output_efficiency_inst{c0},...
                    ' --',batch_header{13},'=',in_heat_rate_instance{c0},...
                    ' --',batch_header{14},'=',out_heat_rate_instance{c0},...
                    ' --',batch_header{15},'=',storage_cap_instance{c0},...
                    ' --',batch_header{16},'=',reg_cost_instance{c0},...
                    ' --',batch_header{17},'=',min_runtime_instance{c0},...
                    ' --',batch_header{18},'=',op_period_instance{c0},...
                    ' --',batch_header{19},'=',int_length_instance{c0},...                    
                    ' --',batch_header{20},'=',lookahead_instance{c0},...
                    ' --',batch_header{21},'=',energy_only_instance{c0},...
                    ' --',batch_header{22},'=',H2_consume_adj_inst{c0},...
                    ' --',batch_header{23},'=',H2_price_instance{c0},...
                    ' --',batch_header{24},'=',H2_use_instance{c0},...
                    ' --',batch_header{25},'=',base_op_instance{c0},...
                    ' --',batch_header{26},'=',NG_price_adj_instance{c0},...
                    ' --',batch_header{27},'=',Renewable_MW_instance{c0},...
                    ' --',batch_header{28},'="',elec_rate_instance{c0},'"',...
                    ' --',batch_header{29},'="',add_param_instance{c0},'"',...
                    ' --',batch_header{30},'="',ren_prof_instance{c0},'"',...                    
                    ' --',batch_header{31},'="',current_int_instance{c0},'"',...                    
                    ' --',batch_header{32},'="',next_int_instance{c0},'"',...
                    ' --',batch_header{33},'="',current_stor_intance{c0},'"',...
                    ' --',batch_header{34},'="',current_max_instance{c0},'"',...
                    ' --',batch_header{35},'="',max_int_instance{c0},'"',...
                    ' --',batch_header{36},'="',outdir,'"',...
                    ' --',batch_header{37},'="',indir,'"',...
                    ' --',batch_header{38},'="',Load_profile,'"',...
                    ' --',batch_header{39},'="',num2str(0),'"',...
                    ];
        fprintf(fileID,'%s\n',GAMS_batch);
        if mod(c0,items_per_batch)==0
            fclose(fileID);
            c2=c2+1;            
            fileID = fopen([dir2,['Batch_Regular_tariffs',num2str(c2),'.bat']],'wt');
        end
    end
    end
    end
    end
    end
    if mod(i0,100)==0
        disp(['File ',num2str(c2),' : ',num2str(i0),' of ',num2str(length(files_tariff2))]);   % Display progress    
    end
end          
fclose(fileID);
disp([num2str(c0),' model runs created']);
