clc
clear

function [inlet_concentration] = distribute_inlet(pressure, Temperature, R, CO2share)
    %ideal Gas, dalton law
    inlet_concentration = [(pressure / (R * Temperature))*CO2share, (pressure/ (R * Temperature))*(1-CO2share), 0];
end


%% Load Parameters
Data;

Channel_H    = 1e-3;                           %Channel height, [m] (-> Denoted as H in the manuscript)
Channel_W   = 1e-2;                           %Channel width, [m] (-> Denoted as W in the manuscript)
Channel_L   = 0.1;  
L_c  = 3e-6;                           %Catalyst layer thickness, [m] (-> Denoted as H_c in the manuscript)

%Gas flow rate 15 mL/min
%v = 0.045833333; %[m/s]

%E_appl = -1.3333333;
%CO2share = 0.6;

v = 3 / 60; %[m/s]

E_appl = -1.25;
CO2share = 0.8;



initial_concentration = distribute_inlet(const.p, const.T, const.R, CO2share);




%Liquid flow rate of 1 ml/min
vL = 1*1e-6/60*1/Channel_H/Channel_W; %[m/s]


[X,FE,y,delP,CD, sol] = channelmodel_full_Ag_Python(E_appl,Channel_L,v,vL,c_int,k,H,BV,const,Channel_H,Channel_W,initial_concentration,por,D,L_c,a, "Heun");

%% Post Processing

%calculate actual Voltage adapted from Baghemihl PhD Dissertation,
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


