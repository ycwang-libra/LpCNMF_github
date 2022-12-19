function [U_final, V_final,objhistory_final] = RSNMF_Multi(X, k, p, m, n, l, options, U, V)
% Robust Structured Nonnegative Matrix Factorization for Image Representation
% RSNMF
% where
%   X
% Notation:
% X ... (mFea x nSmp) data matrix 
%       mFea  ... number of dimensions 
%       nSmp  ... number of samples
% k ... number of classes
% p ...lp·¶Êý
% d ... the dimensionality of each subspace
%%%% r...  r = d x k number of hidden factors/subspace dimensions
% U ... (mFea x r)
% V ... (r x nSmp)

% options ... Structure holding all settings
%
% You only need to provide the above four inputs.
%
% X = UV


differror = options.error;
maxIter = options.maxIter;
nRepeat = options.nRepeat;
minIter = options.minIter - 1;
if ~isempty(maxIter) && maxIter < minIter
    minIter = maxIter;
end
meanFitRatio = options.meanFitRatio;

alp = 0.5 * options.alpha;

r = m * k; 
I_bar = ones(r, l);
O_bar = zeros(r, n - l);
j = 1;
for i=1:m:r
     I_bar(i:(i + m - 1), j:(j + l / k-1)) = zeros(m, l / k);
     j = j + l / k;
end
I = [I_bar O_bar];

Norm = 0;
NormV = 0;

[mFea,nSmp]=size(X);

selectInit = 1;  

if isempty(U)
    U = abs(rand(mFea, r));
    V = abs(rand(r, nSmp));
else
    nRepeat = 1;  
end

[U, V] = NormalizeUV(U, V, NormV, Norm);   

D = eye(nSmp);

if nRepeat == 1
    selectInit = 0;
    minIter = 0;
    if isempty(maxIter)
        objhistory = CalculateObj(X, U, V, I, p, alp);
        meanFit = objhistory*10;  
    else
        if isfield(options,'Converge') && options.Converge
            objhistory = CalculateObj(X, U, V, I, p, alp);
        end
    end
else
    if isfield(options,'Converge') && options.Converge
        error('Not implemented!');
    end
end


tryNo = 0;
nIter = 0;
while tryNo < nRepeat   
    tryNo = tryNo+1;
    maxErr = 1;
    while(maxErr > differror)
        % ===================== update U ========================    
        XDV = X * D * V';
        UVDV = U * V * D * V'; 
        U = U .* (XDV ./ max(UVDV,1e-10));       
        
       % ===================== update V ========================
       UXD = U' * X * D;
       UUVD = U' * U * V * D;
       IV = alp * I .* V;
       DownV = UUVD + IV;
       V = V .* (UXD ./ max(DownV, 1e-10));
        
       % ===================== update D ========================
       Z = X - U * V;
       z = zeros(1,nSmp);
       for i = 1:nSmp
           %z(1,i) = 0.5 * p * (1 ./ (sum(sum(Z(i,:).^(2-p)),2)));
           z(1,i) = 0.5 * p * (1 ./ (sum(sum(Z(:,i).^(2-p)),2)));
       end
       D = diag(z);
        
        nIter = nIter + 1;
        if nIter > minIter
            if selectInit
                objhistory = CalculateObj(X, U, V, I, p, alp);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj(X, U, V, I, p, alp);
                    objhistory = [objhistory newobj];
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;                  
                else
                    if isfield(options,'Converge') && options.Converge
                        newobj = CalculateObj(X, U, V, I, p, alp);
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
        V_final = V; 
        % nIter_final = nIter;
        objhistory_final = objhistory;
    else
       if objhistory(end) < objhistory_final(end)
           U_final = U;
           V_final = V;
           % nIter_final = nIter;
           objhistory_final = objhistory;
       end
    end

    if selectInit
        if tryNo < nRepeat
            %re-start
            U = abs(rand(mFea, r)); %m,r
            V = abs(rand(r, nSmp)); %        
            [U,V] = NormalizeUV(U,V,NormV,Norm);  
            nIter = 0;
        else
            tryNo = tryNo - 1;
            nIter = minIter+1;
            selectInit = 0;
            U = U_final;
            V = V_final;
            objhistory = objhistory_final;
            meanFit = objhistory * 10;
        end
    end
end
[U_final, V_final] = NormalizeUV(U_final, V_final, NormV, Norm);


%==========================================================================
% X, U, V, I, p, alpha
function [obj, dV] = CalculateObj(X, U, V, I, p, alp, deltaVU, dVordU)
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
    
    if mn < MAXARRAY
        dX = U * V - X;
        % obj_NMF = sum(sum(dX.^2));   %||X-UV'||2
        % obj_NMF =        %||X-UV||p,2p --------------------
        obj_NMF = sum(sum(abs(dX).^p),2);
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
            obj_NMF = obj_NMF + sum(sum(abs(dX).^p),2);
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
    IV = I .* V;
    obj_Lap = alp * sum(sum(IV.^2));   %---------
    
    obj = obj_NMF + obj_Lap;
    
function [U, V] = NormalizeUV(U, V, NormV, Norm)
    K = size(U,2);
    if Norm == 2
        if NormV
            norms = max(1e-15,sqrt(sum(V.^2,1)))';
            V = V*spdiags(norms.^-1,0,K,K);
            U = U*spdiags(norms,0,K,K);
        else
            norms = max(1e-15,sqrt(sum(U.^2,1)))';
            U = U*spdiags(norms.^-1,0,K,K);
            V = V*spdiags(norms,0,K,K); 
        end
    else
        if NormV
            norms = max(1e-15,sum(abs(V),1))';
            V = V*spdiags(norms.^-1,0,K,K);
            U = U*spdiags(norms,0,K,K);
        else
            norms = max(1e-15,sum(abs(U),1))';
            U = U*spdiags(norms.^-1,0,K,K);
            %V = V*spdiags(norms,0,K,K);
            V = spdiags(norms,0,K,K)*V;
        end
    end
        