function [U_final, Z_final, F_final, nIter_final, obj_all] = LpCNMF(X, A, r, k, S, W, options, U, Z, F)
% Label Propagation Constrained NMF(LpCNMF)
%
% where
%   X
% Notation:
% X ... (mFea x nSmp) data matrix 
%       mFea  ... number of dimensions 
%       nSmp  ... number of samples
% A/Y ... (nSmp x c)Label matrix of X
% r...  r << min(mFea , nSmp)number of hidden factors/subspace dimensions
% k ... number of classes
% S ... (nSmp x nSmp)diagonal label matrix
% W ... (nSmp x nSmp)weight matrix of the affinity graph 
% U ... (mFea x r) base metric
% Z ... (k x r)  auxiliary metric
% F ... (nSmp x k) Prediction membership matrix


% options ... Structure holding all settings
%
% You only need to provide the above four inputs.
%
% X = U * (FZ)'


if min(min(X)) < 0
    error('Input should be nonnegative!');
end

if ~isfield(options,'error')
    options.error = 1e-5;
end
if ~isfield(options, 'maxIter')
    options.maxIter = [];
end

if ~isfield(options,'nRepeat')
    options.nRepeat = 10;
end

if ~isfield(options,'minIter')
    options.minIter = 30;
end

if ~isfield(options,'meanFitRatio')
    options.meanFitRatio = 0.1;
end

if ~isfield(options,'alpha')
    options.alpha = 10;
end

if isfield(options,'alpha_nSmp') && options.alpha_nSmp
    options.alpha = options.alpha*nSmp;    
end

if isfield(options,'weight') && strcmpi(options.weight,'NCW')
    feaSum = full(sum(X,2));
    D_half = X'*feaSum;
    X = X*spdiags(D_half.^-.5,0,nSmp,nSmp);
end

if ~isfield(options,'Optimization')
    options.Optimization = 'Multiplicative';
end

if ~exist('U','var')
    U = [];
    Z = [];
    F = [];
end

switch lower(options.Optimization)
    case {lower('Multiplicative')} 
        [U_final, Z_final, F_final, nIter_final, obj_all] = LpCNMF_Multi(X, A, r, k, S, W, options, U, Z, F);
    otherwise
        error('optimization method does not exist!');
end


    
        