function [alpha_svm, pred,out] = train(K,K_test,Y,lam)
    % Trains the C-SVM with training data and predicts
    % the presence of a bound on the test dataset.
    % Inputs
    %   K: Gram matrix of the 
    %   K_test: Gram matrix of the test dataset
    %   Y: targets on the training dataset
    %   lam: lambda parameter to train the SVM
    % Outputs
    %   alpha_svm: weights of the support vectors
    %   pred: value of the prediction for each test point
    %   out: value of the learned image for each test point
    
    p = 4;  % length of the subsequence
    n = size(K, 1); 
    C = 1/(2*n*lam);
    % Compute alpha by solving the associated 
    % convex optimization problem
    cvx_begin
        variable alpha_svm(n)
        maximize(2*dot(alpha_svm,Y) - quad_form(alpha_svm, K))
        subject to
            0 <= alpha_svm.*Y <= C
    cvx_end
    % Compute the value of f(x) for each x in the test dataset
    out = ((K_test')*alpha_svm);
    % From that, deduce the predictions in {0;1}
    pred = out>0;
end

