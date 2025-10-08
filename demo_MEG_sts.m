clc;
clear all;
close all;
warning off;
addpath('DLAD');
addpath('MEKT');
addpath(genpath('KTL_DDS'));
addpath(genpath('LDA'));
% Load datasets: 
% 9 subjects, each 22*750*144 (channels*points*trails)
% ALL=[];
% for p = [0.1]
% ALL_acc = [];
% for q = [0.1 0.5 1 10 50 100]
for ds = 1:6
% for ds = 1:6
    % root='FBCSP\MEG-6\';
    root = ['./FBCSP-6/MEG-' num2str(ds) '/'];
    listing=dir([root '*.mat']);
    tic
    % Load data and perform congruent transform+
    fnum=length(listing);
    % Xr=nan(60,200*fnum);
    % Y=nan(200*fnum,1);
    for tr=1:fnum
        % disp(tr);
        % Single target data
        load([root listing(tr).name])
        Xt=x'; Yt=y;  %
        tes=1:fnum; tes(tr)=[];

        for te=1:fnum-1
            % Single source data
            load([root listing(tes(te)).name])
            Xs=x'; Ys=y;
            
            %% RDLAD
            options.r = 10;
            options.beta = 1; % 1
            options.eta = 0.1;
            options.lambda = 10;   %% MEG 10  BCI 5
            options.T = 10;
            
            [Acc,acc_iter,Yt_pred] = RDLAD(Xs,Ys,Xt,Yt,options);
            BCA(tr, te)=Acc;
            classes = unique(Ys);
            [kappa, confusion] = evaluation_measures(Yt, Yt_pred, classes, 'KAPPA');
            KAPPA(tr, te)=kappa;
    
        end
    end
    disp(mean(mean(BCA,1),2)*100')
    ALL_acc(ds,:) = BCA(:);
    ALL_kappa(ds,:) = KAPPA(:);
    
    % time = toc;
    % TIME = [TIME, time];
end
% mean(TIME)
% sqrt(var(TIME))
% mean(ALL_acc)
% ALL = [ALL;ALL_acc];
% end