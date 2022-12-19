function [U_final, V_final, objhistory_final] = RSNMF(X, k, p, m, n, l, options, U, V)
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
% m ... the dimensionality of each subspace
% n ... the number of samples
% l ... the number of labeled samples
% U ... (mFea x r)
% V ... (r x nSmp)
% options ... Structure holding all settings
%
% You only need to provide the above inputs.
%
% X = U * V


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
    options.alpha = 0;
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
    V = [];
end

switch lower(options.Optimization)
    case {lower('Multiplicative')} 
        [U_final, V_final, objhistory_final] = RSNMF_Multi(X, k, p, m, n, l, options, U, V);
    otherwise
        error('optimization method does not exist!');
end


    
        