% This file is voltage control without integrating the duals.
% Find the Matpower manual at matpower4.1\matpower4.1\docs

clear
close all
clc

addpath \matpower4.1\matpower4.1;
addpath(genpath('./'));

% option for the power flow solver. See pag. 82 of the manual.
mpopt=mpoption('OUT_ALL',0,'VERBOSE',0);

% this command sets some useful constants for using MATPOWER. See pag. 12
% of the manual.
define_constants;

%% Testbed
% Other test cases can be found at matpower4.1\matpower4.1 
% This is the test case I used in my paper, a transmission net with 5 buses
% mpc is a structure needed by the solver
mpc = loadcase('case6_Guido');

%% Power Flow

% solve the power flow
mpc = runpf(mpc,mpopt);

% compute voltage phasors (angle are expressed in degrees in the structure mpc, 
% you might need to convert them in radiants to use them, as in the following
% where I compute the voltage phasors)

voltages = mpc.bus(:,VM) .* exp(1i * mpc.bus(:,VA)/180*pi)

% I increase the loads by 10 MW (load buses are 2,3,4). Power must be in MW. 
% Impedances in [p.u.]. See Appendix B of the manual 
mpc.bus(2:4,PD) = mpc.bus(2:4,PD) + [10;10;10];

% I decrease the PV generator (node 5) P output by 5 MW and increase its V output by 0.2 p.u. Heed that Node 5 is
% the second generator in mpc.gen
mpc.gen(2,PG) = mpc.gen(2,PG) - 5;
mpc.gen(2,VG) = mpc.gen(2,VG) + 0.02;

% solve the new power flow and compute new voltage phasors
mpc = runpf(mpc,mpopt);
voltages = mpc.bus(:,VM) .* exp(1i * mpc.bus(:,VA)/180*pi)

%% Optimal Power Flow
%compute the optimal power flow
results = runopf(mpc,mpopt);
voltages_opt =  results.bus(:,VM) .* exp(1i * results.bus(:,VA)/180*pi)
