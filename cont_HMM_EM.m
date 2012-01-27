function [model,log_like,model_kmeans] = cont_HMM_EM(O_all,N)
% learn continuous HMM parameters using EM (one Gaussian)
%
% input:
%   O_all:  1 x seqNum cell, each is a
%               ftDim x T, observed sequence
%   N:      number of hidden states
%
% output:
%   model:      a model, which contains the following estimated parameters
%       A:          N x N, transition matrix, a_ij = Prb(q_j|q_i)
%       mu:         ftDim x N, mean for each Gaussian
%       cov:        ftDim x ftDim x N, covariance matrix for each Gaussian
%       P:          N x 1, prior probabilities
%
%   log_like:   log likelihood of each iteration

seqNum = length(O_all);
ftDim = size(O_all{1},1);
conv_prec = 1e-6;
max_iter = 1000;
lambda = 1e-2;


% initialize Gaussian distributions using kmeans
fprintf('Initializing HMM using kmeans\n')
[model_kmeans] = cont_HMM_kmeans(O_all,N);

P = model_kmeans.P;
A = model_kmeans.A;
est_mu = model_kmeans.mu;
est_cov = model_kmeans.cov;
fprintf('Done kmeans\n')


log_like = zeros(max_iter,1);
for i = 1:max_iter
    new_P = zeros(size(P));
    new_A = zeros(size(A));
    new_mu = zeros(size(est_mu));
    new_cov = zeros(size(est_cov));
    sum_gamma = zeros(N,1);
   
    for seqIdx = 1:seqNum
        O = O_all{seqIdx};
        T = size(O,2);
               
        % precompute emission probabilities
        B = zeros(N,T);
        for state_idx = 1:N
            B(state_idx,:) = mvnpdf(O',est_mu(:,state_idx)',est_cov(:,:,state_idx))';
        end
        
        % compute forward probabilities with precomputed B
        alpha = zeros(N,T);
        scale_alpha = zeros(1,T);
        
        alpha(:,1) = P .* B(:,1);
        scale_alpha(1) = 1./sum(alpha(:,1));
        for t = 2:T
            alpha(:,t) = A'*alpha(:,t-1).*B(:,t);
            scale_alpha(t) = 1./sum(alpha(:,t));
            alpha(:,t) = alpha(:,t) * scale_alpha(t);
        end
                
        % compute backward probabilities
        beta = zeros(N,T);

        beta(:,T) = ones(N,1);
        for t = T-1:-1:1
            beta(:,t) = A*(B(:,t+1).*beta(:,t+1));
            beta(:,t) = beta(:,t) * scale_alpha(t);
        end
        

        % compute posterior probabilities (E-step)
        [gamma] = compGamma(alpha,beta);

        % compute averaged joint posterior (q_i,q_j|O)
        ksi = zeros(N);
        for t = 1:T-1
            ksi_tmp = (alpha(:,t) * (beta(:,t+1).*B(:,t+1))') .* A;
            ksi = ksi + ksi_tmp / sum(sum(ksi_tmp));
        end
        
        % update parameters (M-step)
        new_P = new_P + gamma(:,1);
        new_A = new_A + ksi;
        
        % update continuous Gaussian parameters
        new_mu = new_mu + O*gamma';
        for state_idx = 1:N
            new_cov(:,:,state_idx) = new_cov(:,:,state_idx) + bsxfun(@times,O,gamma(state_idx,:))*O';
        end
        sum_gamma = sum_gamma + sum(gamma,2);
        
        % evaluate log-likelihood
        log_like(i) = log_like(i) - sum(log(scale_alpha)) + ftDim*T*4;  % add a constant to keep it from blowing up
        
    end
    
    % normalize update
%     P = new_P / seqNum;
    P = sum(new_A,2);
    P = P / sum(P);
    A = bsxfun(@times,new_A,1./sum(new_A,2));
    est_mu = bsxfun(@times,new_mu,1./sum_gamma');
    est_cov = bsxfun(@times,new_cov,1./reshape(sum_gamma,1,1,N));
    for state_idx = 1:N
        est_cov(:,:,state_idx) = est_cov(:,:,state_idx) - est_mu(:,state_idx)*est_mu(:,state_idx)' + lambda*eye(ftDim);
        est_cov(:,:,state_idx) = (est_cov(:,:,state_idx) + est_cov(:,:,state_idx)') / 2;
    end
    
    % determine if converged
    if i > 2
        log_like_change = abs(1-log_like(i-1)/log_like(i));
        if log_like_change < conv_prec
            break;      % converged
        end
        
        fprintf('Iteration %i: log_like %f, log_like_change %f\n',i,log_like(i),log_like_change)
    end
end

fprintf('Converged!\n')
model.A = A;
model.P = P;
model.mu = est_mu;
model.cov = est_cov;
log_like = log_like(1:i);