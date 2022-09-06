function [U_final, Z_final,F_final, nIter_final, obj_all] = LpCNMF_Multi(X, A, r, k, S, W, options, U, Z, F)
% Label Propagation Constrained NMF(LpCNMF)
%
% where
%   X
% Notation:
% X ... (mFea x nSmp) data matrix 
%       mFea  ... number of dimensions 
%       nSmp  ... number of samples
% A/Y ... (nSmp x c)Label matrix of X
% r...  r << min(mFea , nSmp) number of hidden factors/subspace dimensions
% k ... number of classes
% S ... (nSmp x nSmp)diagonal label matrix
% W ... (nSmp x nSmp)weight matrix of the affinity graph 
% U ... (mFea x r) base
% Z ... (k x r) auxiliary
% F ... (nSmp x c) membership

% options ... Structure holding all settings
%
% You only need to provide the above four inputs.
%
% X = U*(FZ)'

differror = options.error;
maxIter = options.maxIter;
nRepeat = options.nRepeat;
minIter = options.minIter - 1;
if ~isempty(maxIter) && maxIter < minIter
    minIter = maxIter;
end
meanFitRatio = options.meanFitRatio;

alpha = options.alpha;

Norm = 2;
NormF = 1;

obj_all=[];

[mFea,nSmp]=size(X);

if alpha > 0
    W = alpha * W;
    S = alpha * S;
    
    DCol = full(sum(W,2));
    D = spdiags(DCol,0,nSmp,nSmp);
    L = D - W;
    if isfield(options,'NormW') && options.NormW     
        D_mhalf = spdiags(DCol.^-.5,0,nSmp,nSmp) ;    
        L = D_mhalf * L * D_mhalf;
    end
else
    L = [];
end

selectInit = 1;    
if isempty(U)
    U = abs(rand(mFea, r));
    Z = abs(rand(k, r));
    F = abs(rand(nSmp, k));
else
    nRepeat = 1;  
end

[U, Z, F] = NormalizeUV(U, Z, F, NormF, Norm);   

if nRepeat == 1
    selectInit = 0;
    minIter = 0;
    if isempty(maxIter)
        objhistory = CalculateObj(X, U, Z, F, S, A, L);
        meanFit = objhistory*10;     
    else
        if isfield(options,'Converge') && options.Converge
            objhistory = CalculateObj(X, U, Z, F, S, A, L);
        end
    end
else
    if isfield(options,'Converge') && options.Converge
        error('Not implemented!');
    end
end


tryNo = 0;
nIter = 0;
nCal=0;

while tryNo < nRepeat   
    tryNo = tryNo+1;
    maxErr = 1;
    while maxErr > differror
        % ===================== update U ========================
        XFZ = X * F * Z;
        FF = F' * F;
        UZF = U * Z' * FF * Z;
        U = U .* (XFZ ./ max(UZF,1e-10));        
        
       % ===================== update Z ========================
       FXU = F' * X' * U;
       UU = U' * U;
       FZU = FF * Z * UU;
       Z = Z .* (FXU ./ max(FZU,1e-10));
        
       % ===================== update F ========================   
       XUZ = X' * U * Z';
       WF = W * F;
       SA = S * A;
       Up = XUZ + WF + SA;
       
       FZUZ = F * Z * UU * Z';     
       DF = D * F;
       SF = S * F;     
       Down = FZUZ + DF + SF;
       F = F .* (Up ./ max(Down,1e-10));
        
        nIter = nIter + 1;
        if nCal < maxIter
            if nIter <= maxIter
                obj = CalculateObj(X, U, Z, F, S, A, L);
                obj_all =[obj_all obj];
            end
            nCal = nCal + 1;
        end
        if nIter > minIter
            if selectInit
                objhistory = CalculateObj(X, U, Z, F, S, A, L);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj(X, U, Z, F, S, A, L);
                    objhistory = [objhistory newobj];
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;
                else
                    if isfield(options,'Converge') && options.Converge
                        newobj = CalculateObj(X, U, Z, F, S, A, L);
                        objhistory = [objhistory newobj];
                    end
                    maxErr = 1;
                    if nIter >= maxIter
                        maxErr = 0;
                        if isfield(options,'Converge') && options.Converge
                        else
                            objhistory = 0;
                        end
                    end
                end
            end
        end
    end
    
    if tryNo == 1
        U_final = U;
        Z_final = Z;
        F_final = F;  
        nIter_final = nIter;
        objhistory_final = objhistory;
    else
       if objhistory(end) < objhistory_final(end)
           U_final = U;
           Z_final = Z;
           F_final = F;  
           nIter_final = nIter;
           objhistory_final = objhistory;
       end
    end

    if selectInit
        if tryNo < nRepeat
            %re-start
            U = abs(rand(mFea, r));
            Z = abs(rand(k, r));
            F = abs(rand(nSmp, k));

            [U,Z,F] = NormalizeUV(U,Z,F,NormF,Norm);
        else
            tryNo = tryNo - 1;
            nIter = minIter+1;
            selectInit = 0;
            U = U_final;
            Z = Z_final;
            F = F_final;
            objhistory = objhistory_final;
            meanFit = objhistory * 10;
        end
    end
end

[U_final, Z_final, F_final] = NormalizeUV(U_final, Z_final, F_final, NormF, Norm);

%==========================================================================
% 计算目标函数
function [obj, dV] = CalculateObj(X, U, Z, F, S, A, L, deltaVU, dVordU)
    MAXARRAY = 500*1024*1024/8; % 500M. You can modify this number based on your machine's computational power.
    if ~exist('deltaVU','var')
        deltaVU = 0;
    end
    if ~exist('dVordU','var')
        dVordU = 1;
    end
    dV = [];
    nSmp = size(X,2);
    mn = numel(X);
    nBlock = ceil(mn/MAXARRAY);
    V = F * Z;

    if mn < MAXARRAY
        dX = X - U * V';
        obj_NMF = sum(sum(dX.^2));   %||X-UV'||2
        if deltaVU
            if dVordU
                dV = dX' * U + L * V;
            else
                dV = dX * V;
            end
        end
    else
        obj_NMF = 0;
        if deltaVU
            if dVordU
                dV = zeros(size(V));
            else
                dV = zeros(size(U));
            end
        end
        PatchSize = ceil(nSmp/nBlock);
        for i = 1:nBlock
            if i*PatchSize > nSmp
                smpIdx = (i-1)*PatchSize+1:nSmp;
            else
                smpIdx = (i-1)*PatchSize+1:i*PatchSize;
            end
            dX = U*V(smpIdx,:)'-X(:,smpIdx);
            obj_NMF = obj_NMF + sum(sum(dX.^2));
            if deltaVU
                if dVordU
                    dV(smpIdx,:) = dX'*U;
                else
                    dV = dU+dX*V(smpIdx,:);
                end
            end
        end
        if deltaVU
            if dVordU
                dV = dV + L*V;
            end
        end
    end
    if isempty(L)
        obj_Lap = 0;
    else
        sum1 = sum(sum((F' * L) .* F')); %tr(F'LF)+tr[(F-A)'S(F-A)]
        FA = F - A;
        sum2 = sum(sum(FA' * S .* FA'));
        obj_Lap = sum1 + sum2;      
    end
    obj = obj_NMF + obj_Lap;
    

function [U,Z,F] = NormalizeUV(U,Z,F,NormF,Norm)
    if Norm == 2         
        if NormF 
            norms = max(1e-15,sqrt(sum(F.^2,1)))'; 
            F = F * spdiags(norms.^-1,0,size(F,2),size(F,2));
            normsu = max(1e-15,sqrt(sum(U.^2,1)))';
            U = U * spdiags(normsu.^-1,0,size(U,2),size(U,2));
            Z = spdiags(sqrt(norms),0,size(F,2),size(F,2)) * Z * spdiags(sqrt(normsu),0,size(Z,2),size(Z,2));
        end
    else
        if NormF
            norms = max(1e-15,sum(abs(F),1))';
            F = F * spdiags(norms.^-1,0,size(F,2),size(F,2));
            U = U * spdiags(sqrt(norms),0,size(U,2),size(U,2));
            Z = Z * spdiags(sqrt(norms),0,size(Z,2),size(Z,2));
        end
    end
        