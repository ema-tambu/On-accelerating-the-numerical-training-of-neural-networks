% Using the parareal algorithm to train a Fully Connected Neural Network

% classification task
clear all; close all; clc;

% Load data to train the network
% addpath 'Gradient Descent'\'Classification Task'\
addpath 'parareal_systems'
load X.mat; load y.mat; y = Y;

n = size(X, 1);
d = size(X, 2);
n_classes = 4;

% visualize dataset

figure
scatter(X(:, 1), X(:, 2), [], y);

% transform y to categorical
y_cat = zeros(4, n);
for k = 1:n
    y_cat(y(k)+1,k) = 1;
end

% split the dataset into train and test

split_ratio = 0.3;
k = floor(split_ratio*n);

X_train = X(1:k,:);
X_test = X(k+1:end,:);
y_train = y(1:k);
y_test = y(k+1:end);
y_train_cat = y_cat(:,1:k);
y_test_cat = y_cat(:,k+1:end);

% Define some hyperparameters of the network
sigma = @(t) 1./(1+exp(-t));
sigmaprime = @(t) sigma(t).*(1-sigma(t));

shape = [d, 3, 3, n_classes];

niter = 3e5;
eta = 0.05;

% TRAIN THE NETWORK

[costHistory, W, b] = TrainNetworkParareal( ...
        X_train', y_train_cat, niter, sigma, sigmaprime, eta, shape);
save NNparams.mat W b costHistory
% load NNparams.mat

% DISPLAY RESULTS AND TEST THE NETWORK

% Plot costHistory as done before
figure
plot(linspace(0,niter,niter)', costHistory, '-')
fprintf('Cost Function: %f\n', costHistory(end));

% Test - Predict the class of the points in the test set using the network
y_pred = PredictClasses(W, b, sigma, X_test');
Plot the prediction
figure
scatter(X_train(:, 1), X_train(:, 2), [], y_train, 'o');
hold on;
scatter(X_test(:, 1), X_test(:, 2), [], y_pred, '*');

% confusion matrix
figure
y_test = int32(y_test);
confusionchart(y_test,y_pred)
