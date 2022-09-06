function [U_final, V_final] = SNMFDSR_Multi(X, D, S, r, options, U, V)
% Semi-Supervised NMF With Dissimilarity and Similarity Regularization(SNMFDSR)
%
% where
%   X = U V
% Notation:
% X ... (mFea x nSmp) data matrix 
%       mFea  ... number of dimensions 
%       nSmp  ... number of samples
% D ... (nSmp x nSmp) dissimilarity matrix
% S ... (nSmp x nSmp) similarity matrix
% r...  r << min(mFea , nSmp)number of hidden factors/subspace dimensions

% U ... (mFea x r)
% V ... (r x nSmp)

% options ... Structure holding all settings
%
% You only need to provide the above four inputs.
%
% X = U * V


differror = options.error;
maxIter = options.maxIter;
nRepeat = options.nRepeat;
minIter = options.minIter - 1;
if ~isempty(maxIter) && maxIter < minIter
    minIter = maxIter;
end
meanFitRatio = options.meanFitRatio;

alpha = options.alpha;
belta = 0.01;  %----------------------------------

Norm = 0;  %
NormV = 0;  %

[mFea,nSmp]=size(X);

%A = zeros(nSmp, nSmp);

selectInit = 1;  

if isempty(U)
    U = abs(rand(mFea, r));
    V = abs(rand(r, nSmp));       % r, nSmp
else
    nRepeat = 1;  
end

[U, V] = NormalizeUV(U, V, NormV, Norm);   

%V~ matrix
V_wave = dist(V',V);

if nRepeat == 1
    selectInit = 0;
    minIter = 0;
    if isempty(maxIter)
        objhistory = CalculateObj(X, U, V, D, S, V_wave, alpha, belta); 
        meanFit = objhistory*10;  
    else
        if isfield(options,'Converge') && options.Converge
            objhistory = CalculateObj(X, U, V, D, S, V_wave, alpha, belta);
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
        XV = X*V';  % mnk or pk (p<<mn)
        VV = V*V';  % 
        UVV = U*VV; % 
        U = U.*(XV./max(UVV,1e-10));
        
        % ===================== update V ========================
        UX = U'*X;
        VS = 2*belta*V*S;
        UU = U'*U;
        UUV = UU*V;
        %%% A------------------------
        for i=1:nSmp
            aa(i)=sum(S(i,:));
        end
        size(aa);
        A = diag(aa);
        size(A);
        
        VA = 2*belta*V*A;
        VD = alpha*V*D;
        
        V = V.*((UX + VS)./max((UUV + VA + VD ),1e-10)); 
        
        nIter = nIter + 1;
        if nIter > minIter
            if selectInit
                objhistory = CalculateObj(X, U, V, D, S, V_wave, alpha, belta);
                maxErr = 0;
            else
                if isempty(maxIter)
                    newobj = CalculateObj(X, U, V, D, S, V_wave, alpha, belta);
                    objhistory = [objhistory newobj];
                    meanFit = meanFitRatio*meanFit + (1-meanFitRatio)*newobj;
                    maxErr = (meanFit-newobj)/meanFit;                  
                else
                    if isfield(options,'Converge') && options.Converge
                        newobj = CalculateObj(X, U, V, D, S, V_wave, alpha, belta);
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
        objhistory_final = objhistory;
    else
       if objhistory(end) < objhistory_final(end)
           U_final = U;
           V_final = V;
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
function [obj, dV] = CalculateObj(X, U, V, D, S, V_wave, alpha, belta, deltaVU, dVordU)
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
        obj_NMF = sum(sum(dX.^2));
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
    VV = V' * V;
    DVV = D .* VV;
    obj_Lap1 = alpha * norm(DVV, 1);
    SV = S .* V_wave;
    obj_Lap2 = belta * norm(SV, 1);
    obj = obj_NMF + obj_Lap1 + obj_Lap2;
    
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
            V = spdiags(norms,0,K,K)*V;
        end
    end
        