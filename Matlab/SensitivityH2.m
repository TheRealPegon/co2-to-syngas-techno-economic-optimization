clc
clear

%TLDR: this model cannot solve if H2 inlet not 0
%ASSUMPTIONS:
%inlet is ideal gas (dalton law applies)
%inlet does not contain H2 (as my predecessor did not consider it in the flowsheet either)
%gas pressure 1 bar
%the pressure drop is the liquid electrolye pressure drop
%temperature 25 degrees
%electrolyte flow 1ml/min. Maybe vary this as input?


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


no_sensitivity_samples = 100;
input_data = linspace(0,1, no_sensitivity_samples);
timeout_duration = 240; % 4 minutes in seconds


function [data, maxRes] = try_catch_block(data, i, filename, E_appl, CO2share, timeout_duration, Channel_L, v, vL, c_int, k, H, BV, const, Channel_H, Channel_W, initial_concentration, por, D, L_c, a, solver)
    Data;    
    t_one = tic;

    %This is needed for a timed kill switch
    f = parfeval(@channelmodel_full_Ag_Python,6, E_appl, Channel_L, v, vL, c_int, k, H, BV, const, Channel_H, Channel_W, initial_concentration, por, D, L_c, a, solver);
    maxRes = 0; 
    try
        % Wait for result for max 240 seconds
        [completedIdx, X_tmp, FE_tmp, y_tmp, delP_tmp, CD_tmp, sol] = fetchNext(f, timeout_duration);
        %[X,FE,y,delP,CD, sol] = channelmodel_full_Ag_Python(E_appl,Channel_L,v,vL,c_int,k,H,BV,const,Channel_H,Channel_W,initial_concentration,por,D,L_c,a,solver);
        if isempty(completedIdx)
            % TIMEOUT: The function is still running. KILL IT.
            cancel(f);
            fprintf('i = %d\n', i, "got killed");
            error('Timeout:Exceeded240s', 'Simulation killed after 4 minutes')
        end
        
        % If successful, assign values
        X = X_tmp; FE = FE_tmp; y = y_tmp; delP = delP_tmp; CD = CD_tmp;

        %% Post Processing
        %calculate actual Voltage adapted from Bagemihl PhD Dissertation,
        %Section 4.E
        eta_actA    = const.R*const.T/(0.5*const.F)*asinh(CD/(2*1e-7)); %eq 4.54
        eta_ohm     = CD*(Channel_H/sigma_el+Lm/sigma_m); %eq 4.55
        
        %Potential of OER
        E_anode = 1.23;
        
        eta_tot     = E_anode + eta_actA + BV.ECO + abs(E_appl) + eta_ohm; %eq 4.22
        Vcell = eta_tot;

        error_val = 0;
        maxRes = sol.stats.maxres;

        if sol.stats.maxres > 1e-4
            error_val = 3;
        end
        time = toc(t_one);
    catch e
        % This block triggers if channelmodel fails OR if killed by timeout
        %cancel(f); % Ensure the background task is dead
        fprintf(1,'The identifier was:\n%s',e.identifier);
        fprintf(1,'There was an error! The message was:\n%s',e.message);

        X.tot(1) = NaN; X.het(1) = NaN; X.hom(1) = NaN;
        FE = NaN; y.CO2(:) = NaN; y.C2H4(:) = NaN; y.H2(:) = NaN;
        delP = NaN; CD = NaN; Vcell = NaN;
        
        error_val = 1;
        maxRes = 0;
        time = toc(t_one);
    end

    % Check for unphysical results
    if error_val == 0 && (X.het(1) < 0 || X.hom(1) < 0)
        error_val = 2;
    end
    %this vector represents one calculation
    data_vector = [error_val, time, CO2share, E_appl, v, X.tot(1), X.het(1), X.hom(1), FE, y.CO2(end), y.C2H4(end), y.H2(end), delP, CD, Vcell, maxRes, solver];
    data(i+1, :) = data_vector;
    writematrix(data, append(filename, ".txt")) 
end

for input_dimension = 1:3
    data = strings(no_sensitivity_samples+1, 17);
    data(1,:) = ["Error", "time", "CO2share", "E_appl", "v", "X_tot", "X_het", "X_hom", "FE", "CO2_out", "CO_out", "H2_out", "delP", "CD", "Vcell","maxRes", "Solver"];

    for i = 1:(no_sensitivity_samples)
        fprintf("i = %d\n", i);
        
        %CO2share  = (CO2share_lb + CO2share_ub)/2;
        %E_appl  = (E_appl_lb + E_appl_ub)/2;
        CO2share = 0.8;
        E_appl = -1.4;
        v  = (v_lb + v_ub)/2;
        
        if input_dimension == 1
            CO2share = CO2share_lb + input_data(i)*(CO2share_ub-CO2share_lb);
            filename = "sensitivity_CO2share";
        elseif input_dimension == 2
            E_appl = E_appl_lb + input_data(i)*(E_appl_ub-E_appl_lb);
            filename = "sensitivity_E_appl";
        elseif input_dimension == 3    
            v = v_lb + input_data(i)*(v_ub-v_lb);
            filename = "sensitivity_v";
        end

        initial_concentration = distribute_inlet(const.p, const.T, const.R, CO2share);
        solver = "Heun";
        [data, maxRes] = try_catch_block(data, i, filename, E_appl, CO2share, timeout_duration, Channel_L, v, vL, c_int, k, H, BV, const, Channel_H, Channel_W, initial_concentration, por, D, L_c, a, solver);
        if maxRes > 1e-4
            fprintf("Switching to solver Euler");
            solver = "Euler";
            [data, maxRes] = try_catch_block(data, i, filename, E_appl, CO2share, timeout_duration, Channel_L, v, vL, c_int, k, H, BV, const, Channel_H, Channel_W, initial_concentration, por, D, L_c, a, solver);
        end
    end  
end

toc(t_all)
