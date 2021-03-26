% Initiate the predictions
preds = [];

% === FIRST DATASET ===
% Loads the precomputed Gram matrix of the substring kernel
% for the first training dataset
K = load('gram_matrices/gram_4_substr');
K = K.K;
K = squeeze(K(p,:,:));
% Loads the precomputed Gram matrix of the substring kernel
% for the first test dataset
K_test = load('gram_matrices/test_0.mat');
K_test = K_test.K;
K_test = squeeze(K_test(p,:,1:1000));
% Loads the targets of the first training dataset
Y = readtable('../data/Ytr0.csv');
Y = Y{:,'Bound'};
% Maps targets in {0; 1} to targets in {-1; 1}
Y = 2*Y-1;

% Train the SVM with data from the first training dataset
% and get its predictions on the first test dataset
[alpha_svm,pred,out] = train(K, K_test, Y, 0.1);
% Add the predictions to the list of predictions
preds = [preds;pred]

% === SECOND DATASET ===
% Loads the precomputed Gram matrix of the substring kernel
% for the second training dataset
K = load('gram_matrices/gram_4_substr_1');
K = K.K;
K = squeeze(K(p,:,:));
% Loads the precomputed Gram matrix of the substring kernel
% for the second test dataset
K_test = load('gram_matrices/test_1.mat');
K_test = K_test.K;
K_test = squeeze(K_test(p,:,1:1000));
% Loads the targets of the second training dataset
Y = readtable('../data/Ytr1.csv');
Y = Y{:,'Bound'};
% Maps targets in {0; 1} to targets in {-1; 1}
Y = 2*Y-1;

% Train the SVM with data from the second training dataset
% and get its predictions on the second test dataset
[alpha_svm,pred,out] = train(K, K_test, Y, 0.1);
% Add the predictions to the list of predictions
preds = [preds;pred]

% === THIRD DATASET ===
% Loads the precomputed Gram matrix of the substring kernel
% for the third training dataset
K = load('gram_matrices/gram_4_substr_2');
K = K.K;
K = squeeze(K(p,:,:));
% Loads the precomputed Gram matrix of the substring kernel
% for the third test dataset
K_test = load('gram_matrices/test_2.mat');
K_test = K_test.K;
K_test = squeeze(K_test(p,:,1:1000));
% Loads the targets of the third training dataset
Y = readtable('../data/Ytr2.csv');
Y = Y{:,'Bound'};
% Maps targets in {0; 1} to targets in {-1; 1}
Y = 2*Y-1;

% Train the SVM with data from the third training dataset
% and get its predictions on the third test dataset
[alpha_svm,pred,out] = train(K, K_test, Y, 0.1);
% Add the predictions to the list of predictions
preds = [preds;pred]


% === SAVING THE PREDICTIONS ===
% Add the test indexes and write the table in the correct format
preds = [(0:2999)' preds];
T = array2table(preds);
T.Properties.VariableNames(1:2) = {'Id','Bound'}
writetable(T,'test_predictions.csv')

% f(x) = K(x,x_i)@alpha

% preds = [];

% K = load('gram_4_substr');
% K = K.K;
% K = squeeze(K(p,:,:));
% split = 1500;
% K_train = K(1:split,1:split);
% K_val = K(1:split,split+1:end);
% Y = readtable('data/Ytr0.csv');
% Y = Y{:,'Bound'};
% Y_train = Y(1:split);
% Y_val = Y(split+1:end);
% [alpha_svm, out, pred] = train(K_train, K_val, (Y_train*2)-1, 0.1);
% sum(out==Y_val)


