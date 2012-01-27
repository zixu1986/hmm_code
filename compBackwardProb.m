function [beta] = compBackwardProb(O,A,B,scale_alpha)
% compute backward probabilities
%
% input:
%   O:          1 x T, sequence
%   A:          N x N, transition matrix, a_ij = Prb(q_j|q_i)
%   B:          N x M, emission matrix, b_ij = Prb(o_j|q_i)
%   scale_alpha:    T, a series of scalars to make alpha in range
%
% output:
%   beta:       N x T, backward probabilities
%                   P(O_{t+1,..,T}|Q_t=i) for beta(i,t)

[N M] = size(B);
T = length(O);
beta = zeros(N,T);

beta(:,T) = ones(N,1);
for t = T-1:-1:1
    beta(:,t) = A*(B(:,O(t+1)).*beta(:,t+1));
    beta(:,t) = beta(:,t) * scale_alpha(t);
end