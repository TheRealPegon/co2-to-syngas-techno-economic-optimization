clc
clear

%ASSUMPTIONS:
%inlet is ideal gas (dalton law applies)
%inlet does not contain H2 (as my predecessor did not consider it in the flowsheet either)
%gas pressure 1 bar
%the pressure drop is the liquid electrolye pressure drop
%temperature 25 degrees
%electrolyte flow 1ml/min. Maybe vary this as input?
%gas flow 15ml/min. Maybe vary this as input?
%Other geometries as in original Bagemihl paper


%500 datapoints on Ryzen 5800X take about 3 hours per worker
rng default % For reproducibility


function [inlet_concentration] = distribute_inlet(pressure, Temperature, R, CO2share)
    %ideal Gas, dalton law
    inlet_concentration = [(pressure / (R * Temperature))*CO2share, (pressure/ (R * Temperature))*(1-CO2share), 0];
end


t_all =tic;
%% Load Parameters
Data;

Channel_H    = 1e-3;                           %Channel height, [m] (-> Denoted as H in the manuscript)
Channel_W   = 1e-2;                           %Channel width, [m] (-> Denoted as W in the manuscript)
Channel_L   = 0.1;  
L_c  = 3e-6;                           %Catalyst layer thickness, [m] (-> Denoted as H_c in the manuscript)

%Liquid flow rate of 1 ml/min
vL = 1*1e-6/60*1/Channel_H/Channel_W; %[m/s]

E_appl_ub = -0.5;
E_appl_lb = -1.4;

CO2share_lb = 0.2;
CO2share_ub = 1;

v_lb = 5*1e-6/60*1/Channel_H/Channel_W;
v_ub = 50*1e-6/60*1/Channel_H/Channel_W;

no_samples = 2000;
no_sensitivity_samples = 100;
parallelagents = 4;
projected_time = 40*no_samples/3600;
%fprintf("Projected time to completion: ", projected_time, "hours", "\n")
timeout_duration = 240; % 4 minutes in seconds

%0 = LHS, 1 = random
generation_type = 0;

if generation_type == 0 
    LHS = lhsdesign(no_samples, 3);
elseif generation_type == 1
    LHS = rand(no_samples, 3);
elseif generation_type == 2
    LHS = linspace(0,1, no_sensitivity_samples);
    LHS = LHS(2);
    LHS = LHS(3);
end

%for worker=2:2
parfor (worker=1:parallelagents, parallelagents)
    
    data = strings(no_samples/parallelagents+1, 17);
    data(1,:) = ["Error", "time", "CO2share", "E_appl", "v", "X_tot", "X_het", "X_hom", "FE", "CO2_out", "CO_out", "H2_out", "delP", "CD", "Vcell", "maxRes", "Solver"];
    filename = append("worker", string(worker));
    
    %for i = 1:500
    for i = 1:(no_samples/parallelagents)
        fprintf("i = %d worker = %s\n", i, string(worker));
        sample = i*worker;
        CO2share = CO2share_lb + LHS(sample, 1)*(CO2share_ub-CO2share_lb);
        E_appl = E_appl_lb + LHS(sample, 2)*(E_appl_ub-E_appl_lb);
        v = v_lb + LHS(sample, 3)*(v_ub-v_lb);
        
        initial_concentration = distribute_inlet(const.p, const.T, const.R, CO2share);
        solver = "Heun";
        [data, maxRes] = try_catch_block(data, i, filename, E_appl, CO2share, timeout_duration, Channel_L, v, vL, c_int, k, H, BV, const, Channel_H, Channel_W, initial_concentration, por, D, L_c, a, solver);
        if maxRes > 1e-4
            fprintf("Switching to solver Euler for: i = %d worker = %s\n", i, string(worker));
            solver = "Euler";
            [data, maxRes] = try_catch_block(data, i, filename, E_appl, CO2share, timeout_duration, Channel_L, v, vL, c_int, k, H, BV, const, Channel_H, Channel_W, initial_concentration, por, D, L_c, a, solver);
             fprintf("DONE solver Euler for: i = %d worker = %s\n", i, string(worker));
        end
    end
end

toc(t_all)
