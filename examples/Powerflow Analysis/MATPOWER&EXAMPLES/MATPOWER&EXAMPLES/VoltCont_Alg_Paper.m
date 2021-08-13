% This file is voltage control without integrating the duals.
clear
close all
clc

addpath matpower4.1\matpower4.1;
addpath(genpath('./'));
mpopt=mpoption('OUT_ALL',0,'VERBOSE',0);

define_constants;

%% Testbed
mpc = loadcase('case6');
num_bus = 5;

L=[0 1 0 0 1;
   1 0 1 0 0;
   0 1 0 1 0;
   0 0 1 0 0;
   1 0 0 0 0];

Vmin = 0.94;
VMAX = 1.02;

% generators and loads
gen = [1;5];
num_gen = 2;
loads = [2;3;4];
num_loads = 3;

% control limits
x_max=[VMAX;10;10;10;VMAX];
x_min=[0.97;-20;-20;-20;0.97];

% fixing rand seed
rand('seed',28)

% augmenting reactive power demand to have a voltage violation
mpc.bus(:,QD) = mpc.bus(:,QD)*1.6.*rand(5,1);

%% Alg parameters
T = 1000;

lamb_M = zeros(5,T);
%lamb_m(:,1)=0.0005*rand(5,1);

volt = zeros(5,T);
x = zeros(5,T);
x(1,1) = 1;
x(5,1) = 1;
xtil = x;

% setting alg parameters
eta1=2*[1;1;1;10;2];
eta2 = 0.1*[1;10;10;10;10];
eta3 = 0.1*[10;1;1;1;1];

% vectors of function values
F_t = zeros(5,T);

% running the first PF to have initial conditions
mpc = runpf(mpc,mpopt);
volt(:,1) = mpc.bus(:,VM);

for tt = 2:T
% value of f and g
    f_t = zeros(5,1);
    
    for ll = 1:num_loads
        if volt(loads(ll),tt-1) < Vmin
            f_t(loads(ll)) = Vmin - volt(loads(ll),tt-1); % positive value
        end
    end
    
    F_t(:,tt) = f_t;
    
% Computation of control inputs    
    xtil(:,tt)=x(:,tt-1)+diag(eta1)*(f_t + diag(eta2)*L*(lamb_M(:,tt-1)));
    
% Lagrange Multipliers-like update    
    lamb_M(:,tt) = max(0,diag(eta3)*(xtil(:,tt) - x_max));

% Projection
    x(:,tt) = xtil(:,tt);
    
    for nn = 1:num_bus
        if x(nn,tt) > x_max(nn)
            x(nn,tt) = x_max(nn);
        end
        if x(nn,tt) < x_min(nn)
            x(nn,tt) = x_min(nn);
        end
    end
    
% Actuation
    for ll = 1:num_loads
        mpc.gen(loads(ll),QG) = x(loads(ll),tt);
    end
    
    for gg = 1:num_gen
        mpc.gen(gen(gg),VG) = x(gen(gg),tt);
    end
    
    mpc = runpf(mpc,mpopt);
    volt(:,tt) = mpc.bus(:,VM);
end

figure
plot(volt(1,:),'b')
hold on
plot(volt(2,:),'r')
plot(volt(3,:),'r')
plot(volt(4,:),'r')
plot(volt(5,:),'b')
% 
% figure
% plot(F_t')
% 
% figure
% plot(lamb_M')

legend('generators','loads')
ylabel('Voltages')

contr_max = x_max;
contr_max(1,1) = VMAX-1;
contr_max(end,1) = VMAX-1;
scale = [-1;0;0;0;-1]*ones(1,T);
contr_eff = (x+scale)./(contr_max*ones(1,T));

figure
plot(contr_eff')

legend('bus 1','bus 2','bus 3','bus 4','bus 5')

figure
plot(volt')
legend('bus 1','bus 2','bus 3','bus 4','bus 5')