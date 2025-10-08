function [max_acc,acc_iter,Cls] = RDLAD(X_src,Y_src,X_tar,Y_tar,options)
% global options
[m,ns] = size(X_src);
nt = size(X_tar,2);
Ys = bsxfun(@eq, Y_src(:), 1:max(Y_src));
Ys = Ys';
n = ns+nt;
C = size(Ys,1);
X = [X_src X_tar];
X = X*diag(sparse(1./sqrt(sum(X.^2))));

% Construct kernel
K = kernel_meda('sam',X,sqrt(sum(sum(X .^ 2).^0.5)/(n + m)));
Ks = K(:,1:ns);
Kt = K(:,ns+1:end);
Beta = (Ks*Ks'+0.1*eye(n))\(Ks*Ys');
Yt_pred = Beta'*Kt;
[~,Cls] = max(Yt_pred',[],2);
acc = mean(Y_tar == Cls);

T = eye(C); %label encoding matrix

for c = 1:C
    nsc(c) = length(find(Y_src == c));
end
max_acc = 0;
acc_iter = [];
for iter = 1:options.T
    % Compute coefficients vector W and Ypred via SLR
    % Compute P
    for j=1:C
        v = Yt_pred-repmat(T(:,j),1,nt);
        q(:,j)=sum(v'.*v',2);
    end
    if options.r==1
        P = zeros(nt,C);
        [~,idx] = min(q,[],2);
        for i = 1:nt
            P(i,idx(i)) = 1;
        end
    else
        %         mm = (options.r.*q).^(1/(1-options.r-eps));
        %         s0 = sum(mm,2);
        %         P = mm./repmat(s0,1,C);
        P = q.^(1/(1-options.r-eps))./repmat(sum(q.^(1/(1-options.r-eps)),2),1,C);
    end
    F = (P.^options.r)';
    S = diag(sum(F,1));
    
    % Construct Hard MMD matrix
    % [~,ClsCls] = max(P,[],2);
    % PP = bsxfun(@eq, ClsCls(:), 1:max(ClsCls));
    % ntc = sum(PP,1);
    % e = [1 / ns * ones(ns,1); -1 / nt * ones(nt,1)];
    % G1 = e * e' * C;
    % Ns = diag(nsc)^-1;
    % Nt = diag(ntc)^-1;
    % % Nt(find(Nt==NaN))=0;
    % Nt(find(Nt==Inf))=0;
    % if size(PP,2)==1
    %     PP = [PP,PP];
    % end
    % G2 = [Ys'*(Ns*Ns)*Ys,-Ys'*(Ns*Nt)*PP';-PP*(Nt*Ns)*Ys,PP*(Nt*Nt)*PP'];
    % G = G1 + G2;
    % G = G / norm(G,'fro');
    
    % Construct Soft MMD matrix
    ntc = sum(P,1);
    e = [1 / ns * ones(ns,1); -1 / nt * ones(nt,1)];
    G1 = e * e' * C;
    Ns = diag(nsc)^-1;
    Nt = diag(ntc)^-1;
    G2 = [Ys'*(Ns*Ns)*Ys,-Ys'*(Ns*Nt)*P';-P*(Nt*Ns)*Ys,P*(Nt*Nt)*P'];
    G = G1 + G2;
    G = G / norm(G,'fro');
    

    % For discriminability
    rs=1/ns*onehot(Ys,unique(Y_src)); rt=zeros(nt,C);
    if iter > 1
        rt=1/nt*onehot(Cls,unique(Y_src)); 
    end
    Ms=[]; Mt=[];
    for i=1:C
        Ms=[Ms,repmat(rs(:,i),1,C-1)];
        idx=1:C; idx(i)=[];
        Mt=[Mt,rt(:,idx)];
    end
    Rmax=[Ms*Ms',-Ms*Mt';-Mt*Ms',Mt*Mt'];
    Rmax = Rmax / norm(Rmax,'fro');
    
    %Compute Beta
    A = blkdiag(eye(ns),S^0.5);
    Y = [Ys,F*S^-1];
    Beta = ((A * A + options.lambda * G + options.beta * Rmax) * K + options.eta * speye(n,n)) \ ( A * A * Y');
    Yt_pred = Beta' * Kt;
    [~,Cls] = max(Yt_pred',[],2);
    
    %% Compute accuracy
    Acc = mean(Y_tar == Cls);
    acc_iter = [acc_iter;Acc];
    if Acc >= max_acc
        max_acc = Acc;
    else
        break;
    end
    %fprintf('Iteration:%02d, Acc=%f\n',iter,Acc);

end
end


function y_onehot=onehot(y,class)
    % Encode label to onehot form
    % Input:
    % y: label vector, N*1
    % Output:
    % y_onehot: onehot label matrix, N*C

    nc=length(class);
    y_onehot=zeros(length(y), nc);
    for i=1:length(y)
        y_onehot(i, class==y(i))=1;
    end
end
