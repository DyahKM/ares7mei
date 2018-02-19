addpath('HFusion')
addpath('AresFilter')
addpath('data')
clc;clear;

% Load data. 
% smallpox is the California Smallpox Data.
% measel is the New York measle data.
load Tycho
events = events_smallpox;

% Parameter. 
% Lambdas is the H-FUSION Lagrangian multiplier. 
% alpha is the ratio of smoothness and periodicity.
% gama is the parameter of Cost function in Iterative method.
% threshold is the cut-off in Fourier Filter.
% L is the Annilhilating length.
% iterationTime is the constant iteration time without stop criteria.
lambdas = 1;
alpha = 0.5;
gama = 0.5;
iterationTime = 4;


% set the report configuration, config_rep_dur is RD(report duration),
% config_rep_over is shift.
config_rep_dur =1:2:60;
config_rep_over =1:1:26;
xdim = length(config_rep_dur);
ydim = length(config_rep_over);

% First Phase: reconstruct sequence by H-Fusion.
[Out, Out_LSQ] = hfusion(events, lambdas, alpha, config_rep_dur, config_rep_over);


% Second Phase: reconstruct sequence by Annilhilating
% Inc_A is the improvement for each iteration
% Out_A1 is the result for first iteration
% Out_AL is the result for the last iteration
% All the result will be saved in a matfile
stops = [1:1:20];
RMSE = zeros(20,5);

for i = 14
    [Inc_A,Out_A1,Out_AL] = iteration(Out, events, iterationTime, gama, i);
    file_name = strcat('Experiment-stopRatio-',num2str(stops(i)),'-last.mat');
    m=matfile(file_name,'writable',true);
    m.Inc_A = Inc_A;
    m.Out_A1 = Out_A1;
    m.Out_AL = Out_AL;
    ArgRMSE = mean(Inc_A.ActError);
    RMSE(i,:)= ArgRMSE;
end



