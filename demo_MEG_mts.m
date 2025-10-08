clc;
clear all;
close all;
warning off;
addpath('DLAD');
addpath(genpath('MEKT'));
addpath(genpath('KTL_DDS'));
addpath(genpath('LDA'));
% ALL = [];
% for p = [0.001 0.01 0.1 1 10 100]
ALL_acc = [];
% for q = [0.1 0.5 1 10 50 100]
for ds = 1:6
    % Load datasets: 
    % 9 subjects, each 22*750*144 (channels*points*trails)
    % root='MuAlphaBeta\MEG-1\';
    root = ['./FBCSP-6/MEG-' num2str(ds) '/'];
    listing=dir([root '*.mat']);

    % Load data and perform congruent transform+
    fnum=length(listing);
    Xr=nan(60,200*fnum); % 60 18
    Y=nan(200*fnum,1);
    

    for f=1:fnum
        load([root listing(f).name])
        idf=(f-1)*200+1:f*200;
    
        Y(idf) = y'; 
        Xr(:,idf) = x';
    end
    
    BCA=zeros(fnum,1);
    KAPPA=zeros(fnum,1);
    for n=1:fnum
        % disp(n)
        % Single target data & multi source data
        idt=(n-1)*200+1:n*200;
        ids=1:200*fnum; ids(idt)=[];             
        Yt=Y(idt); Ys=Y(ids);
        idsP=Yt==2; idsN=Yt==1;
        Xt=Xr(:,idt);  Xs=Xr(:,ids);

        %% RDLAD
        options.r = 10;
        options.beta = 1; % 1
        options.eta = 0.1;
        options.lambda = 10; % 10
        options.T = 10;
        
        [Acc,acc_iter,Yt_pred] = RDLAD(Xs,Ys,Xt,Yt,options);
        classes = unique(Ys);
        [kappa, confusion] = evaluation_measures(Yt, Yt_pred, classes, 'KAPPA');
        BCA(n)=Acc;
        KAPPA(n)=kappa;
        
    end
    disp(mean(BCA)*100)
    ALL_acc(ds,:) = BCA;
    ALL_kappa(ds,:) = KAPPA;
end
%     ALL = [ALL;ALL_acc];
% end
