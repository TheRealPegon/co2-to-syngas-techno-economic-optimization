function [results] = M3_CO_wrapper(E_appl, initial_concentration)
    %% Load Parameters
    Data;
    
    Channel_H    = 1e-3;                           %Channel height, [m] (-> Denoted as H in the manuscript)
    Channel_W   = 1e-2;                           %Channel width, [m] (-> Denoted as W in the manuscript)
    Channel_L   = 0.01;  
    L_c  = 3e-6;                           %Catalyst layer thickness, [m] (-> Denoted as H_c in the manuscript)
    
    %Gas flow rate 15 mL/min
    v = 15*1e-6/60*1/Channel_H/Channel_W; %[m/s]
    
    %Liquid flow rate of 1 ml/min
    vL = 1*1e-6/60*1/Channel_H/Channel_W; %[m/s]
    
    [X,FE,y,delP,CD] = channelmodel_full_Ag_Python(E_appl,Channel_L,v,vL,c_int,k,H,BV,const,Channel_H,Channel_W,initial_concentration,por,D,L_c,a);
    
    %% Post Processing

    %calculate actual Voltage adapted from Bagemihl PhD Dissertation,
    %Section 4.E
    eta.actA    = const.R*const.T/(0.5*const.F)*asinh(CD/(2*1e-7)); %eq 4.54
    eta.ohm     = CD*(Channel_H/sigma_el+Lm/sigma_m); %eq 4.55
    
    %Potential of OER
    E_anode = 1.23;

    eta.tot     = E_anode + eta.actA + BV.ECO + abs(E_appl) + eta.ohm; %eq 4.22
    Vcell = eta.tot;

    results.X = X;
    results.FE = FE;
    results.y = y;
    results.delP = delP;
    results.CD = CD;
    results.Vcell = Vcell;
end

