clc
clear

rng default

function [inlet_concentration] = distribute_inlet(pressure, Temperature, R, CO2share, H2share)
    %ideal Gas, dalton law
    % CO share is whatever remains after CO2 and H2
    COshare = 1 - CO2share - H2share;
    if COshare < 0
        error('CO2share + H2share cannot exceed 1');
    end
    inlet_concentration = [(pressure / (R * Temperature))*CO2share, ...  % CO2
                           (pressure / (R * Temperature))*COshare, ...   % CO
                           (pressure / (R * Temperature))*H2share];      % H2
end

t_all = tic;
%% Load Parameters
Data;

Channel_H = 1e-3;
Channel_W = 1e-2;
Channel_L = 0.1;
L_c       = 3e-6;

vL = 1*1e-6/60*1/Channel_H/Channel_W;

% Fix all variables at midpoint for H2 sensitivity study
CO2share_fixed = 0.8;
E_appl_fixed   = -1.4;
v_fixed        = (5*1e-6/60*1/Channel_H/Channel_W + 50*1e-6/60*1/Channel_H/Channel_W) / 2;

% H2 share range - upper bound must respect CO2share_fixed
% i.e. H2share_ub + CO2share_fixed <= 1, so max H2share = 0.2
H2share_lb = 0.0;
H2share_ub = 1 - CO2share_fixed;  % = 0.2, leaves room for CO

no_sensitivity_samples = 100;
input_data  = linspace(0, 1, no_sensitivity_samples);
timeout_duration = 240;

function [data, maxRes] = try_catch_block(data, i, filename, E_appl, CO2share, H2share, timeout_duration, Channel_L, v, vL, c_int, k, H, BV, const, Channel_H, Channel_W, initial_concentration, por, D, L_c, a, solver)
    Data;
    t_one = tic;

    f = parfeval(@channelmodel_full_Ag_Python, 6, E_appl, Channel_L, v, vL, c_int, k, H, BV, const, Channel_H, Channel_W, initial_concentration, por, D, L_c, a, solver);
    maxRes = 0;
    try
        [completedIdx, X_tmp, FE_tmp, y_tmp, delP_tmp, CD_tmp, sol] = fetchNext(f, timeout_duration);
        if isempty(completedIdx)
            cancel(f);
            fprintf('i = %d got killed\n', i);
            error('Timeout:Exceeded240s', 'Simulation killed after 4 minutes')
        end

        X = X_tmp; FE = FE_tmp; y = y_tmp; delP = delP_tmp; CD = CD_tmp;

        eta_actA = const.R*const.T/(0.5*const.F)*asinh(CD/(2*1e-7));
        eta_ohm  = CD*(Channel_H/sigma_el + Lm/sigma_m);
        E_anode  = 1.23;
        eta_tot  = E_anode + eta_actA + BV.ECO + abs(E_appl) + eta_ohm;
        Vcell    = eta_tot;

        error_val = 0;
        maxRes    = sol.stats.maxres;
        if sol.stats.maxres > 1e-4
            error_val = 3;
        end
        time = toc(t_one);

    catch e
        fprintf(1,'The identifier was:\n%s', e.identifier);
        fprintf(1,'There was an error! The message was:\n%s', e.message);

        X.tot(1) = NaN; X.het(1) = NaN; X.hom(1) = NaN;
        FE = NaN; y.CO2(:) = NaN; y.C2H4(:) = NaN; y.H2(:) = NaN;
        delP = NaN; CD = NaN; Vcell = NaN;

        error_val = 1;
        maxRes    = 0;
        time      = toc(t_one);
    end

    if error_val == 0 && (X.het(1) < 0 || X.hom(1) < 0)
        error_val = 2;
    end

    % Added H2share to data vector
    data_vector = [error_val, time, CO2share, H2share, E_appl, v, ...
                   X.tot(1), X.het(1), X.hom(1), FE, ...
                   y.CO2(end), y.C2H4(end), y.H2(end), ...
                   delP, CD, Vcell, maxRes, solver];
    data(i+1, :) = data_vector;
    writematrix(data, append(filename, ".txt"))
end

%% H2 sensitivity study
data = strings(no_sensitivity_samples+1, 18);
data(1,:) = ["Error", "time", "CO2share", "H2share", "E_appl", "v", ...
             "X_tot", "X_het", "X_hom", "FE", ...
             "CO2_out", "CO_out", "H2_out", ...
             "delP", "CD", "Vcell", "maxRes", "Solver"];

filename = "sensitivity_H2share";

for i = 1:no_sensitivity_samples
    fprintf("i = %d\n", i);

    H2share = H2share_lb + input_data(i)*(H2share_ub - H2share_lb);

    initial_concentration = distribute_inlet(const.p, const.T, const.R, CO2share_fixed, H2share);

    solver = "Heun";
    [data, maxRes] = try_catch_block(data, i, filename, E_appl_fixed, CO2share_fixed, H2share, ...
                                     timeout_duration, Channel_L, v_fixed, vL, c_int, k, H, BV, const, ...
                                     Channel_H, Channel_W, initial_concentration, por, D, L_c, a, solver);

    if maxRes > 1e-4
        fprintf("Switching to solver Euler\n");
        solver = "Euler";
        [data, maxRes] = try_catch_block(data, i, filename, E_appl_fixed, CO2share_fixed, H2share, ...
                                         timeout_duration, Channel_L, v_fixed, vL, c_int, k, H, BV, const, ...
                                         Channel_H, Channel_W, initial_concentration, por, D, L_c, a, solver);
    end
end

toc(t_all)