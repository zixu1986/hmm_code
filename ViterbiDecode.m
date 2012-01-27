function [decode_seq] = ViterbiDecode(O,A,B,P)
% do Viterbi decoding
%
% input:
%   O:          1 x T, sequence
%   A:          N x N, transition matrix, a_ij = Prb(q_j|q_i)
%   B:          N x M, emission matrix, b_ij = Prb(o_j|q_i)
%   P:          N x 1, prior probabilities
%
% output:
%   decode_seq: 1 x T, sequence

[N M] = size(B);
T = length(O);

decode_seq = zeros(1,T);
score = zeros(N,T);     % log-prob of sequences
max_state = zeros(N,T); % records maximum state at each time stamp

score(:,1) = log(P) + log(B(:,O(1)));
for t = 2:T
    [score(:,t),max_state(:,t)] = max(bsxfun(@plus,A',score(:,t-1)'),[],2);
    score(:,t) = score(:,t) + log(B(:,O(t)));
end

% back tracking
[~,decode_seq(T)] = max(score(:,T));
for t = T-1:-1:1
    decode_seq(t) = max_state(decode_seq(t+1),t+1);
end