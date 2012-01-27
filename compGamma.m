function [gamma] = compGamma(alpha,beta)
% compute gamma, the posterior of hidden states given observation
%
% input:
%   alpha:      N x T, forward probabilities
%                   P(O_{1..t},Q_t=i) for alpha(i,t)
%   beta:       N x T, backward probabilities
%                   P(O_{t+1,..,T}|Q_t=i) for beta(i,t)
%       alpha and beta are scaled using the same scale_factors
%
% output:
%   gamma:      N x T, posterior probabilities of hidden states
%                   P(Q_t=i|O) for gamma(i,t)

gamma = alpha.*beta;
gamma = bsxfun(@times,gamma,1./sum(gamma));