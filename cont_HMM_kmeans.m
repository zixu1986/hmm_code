function [model] = cont_HMM_kmeans(O_all,N)

lambda = 1e-3;      % regularize the diagonal entries

% convert O_all cell array into X (continuous array) and indices (first,second)
X = [];
fst_idx = logical([]);
snd_idx = logical([]);
for i = 1:length(O_all)
    X = [X O_all{i}];
    fst_idx = [fst_idx true(1,size(O_all{i},2)-1) false];
    snd_idx = [snd_idx false true(1,size(O_all{i},2)-1)];
end
ftDim = size(X,1);


% do kmeans 20 times and return the best clustering result
opt_clst = kmeans(X',N,'emptyaction','singleton','replicates',20);

model.mu = zeros(ftDim,N);
model.cov = zeros(ftDim,ftDim,N);
% estimate hidden state observation probability
for state_i = 1:N
    clst_idx = opt_clst==state_i;
    model.mu(:,state_i) = mean(X(:,clst_idx),2);
    model.cov(:,:,state_i) = cov(X(:,clst_idx)')+lambda*eye(ftDim);
end

model.A = zeros(N,N);
% estimate transition probability
for state_i = 1:N
    for state_j = 1:N
        model.A(state_i,state_j) = sum((opt_clst(fst_idx)==state_i) & (opt_clst(snd_idx)==state_j));
    end
end
model.P = sum(model.A,2);
model.A = bsxfun(@times,model.A,1./sum(model.A,2));
model.P = model.P / sum(model.P);
model.pi = histc(opt_clst,unique(opt_clst));
model.pi = model.pi / sum(model.pi);