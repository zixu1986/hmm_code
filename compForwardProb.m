function [alpha,scale_alpha] = compForwardProb(O,A,B,P)
% compute forward probabilities
%
% input:
%   O:          1 x T, sequence
%   A:          N x N, transition matrix, a_ij = Prb(q_j|q_i)
%   B:          N x M, emission matrix, b_ij = Prb(o_j|q_i)
%   P:          N x 1, prior probabilities
%
% output:
%   alpha:      N x T, forward probabilities
%                   P(O_{1..t},Q_t=i) for alpha(i,t)
%   scale_alpha:    T, a series of scalars to make alpha in range

[N M] = size(B);
T = length(O);
alpha = zeros(N,T);
scale_alpha = zeros(1,T);

alpha(:,1) = P.*B(:,O(1));
scale_alpha(1) = 1./sum(alpha(:,1));
for t = 2:T
    alpha(:,t) = A'*alpha(:,t-1).*B(:,O(t));
    scale_alpha(t) = 1./sum(alpha(:,t));
    alpha(:,t) = alpha(:,t) * scale_alpha(t);
end
