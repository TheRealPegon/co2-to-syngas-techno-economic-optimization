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