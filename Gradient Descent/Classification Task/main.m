%% 1)
% This code reproduces the result of the paper 
% "Deep Learning: An Introduction for Applied Mathematicians"
% by training a Feed Forward NN with arbitrary shape, sigmoid activation 
% function and using as training data the very same data of the paper.
clear all; close all; clc;

% visualize data (in this easy example)
x1 = [0.1,0.3,0.1,0.6,0.4];
y1 = [0.1,0.4,0.5,0.9,0.2];

x2 = [0.6,0.5,0.9,0.4,0.7];
y2 = [0.3,0.6,0.2,0.4,0.6];

scatter(x1, y1, 'blue')
hold on;
scatter(x2, y2, 'red')

x = [x1, x2; y1, y2];
y = [ones(1,5) zeros(1,5); zeros(1,5) ones(1,5)];

% define activation functions
sigma = @(t) 1./(1+exp(-t));
sigmaprime = @(t) sigma(t).*(1-sigma(t));

shape = [2, 3, 3, 2];   % ANY NETWORK SHAPE (this includes
%                           input & output layers too)
niter = 3e5;
eta = 0.05;

[costHistory, W, b] = GradientDescentClassification( ...
        x, y, niter, sigma, sigmaprime, eta, shape);

save NNparams.mat W b

% plot results
figure
plot(linspace(0,niter,niter)', costHistory, '-')
fprintf('Cost Function: %f\n', costHistory(end));

% prediction
load NNparams.mat
% generate new random data in the square [0,1]^2
n = 300;
testDatax = rand(n, 2);
testDatay = PredictClasses(W, b, sigma, testDatax');

x = x';
y = int32([zeros(5, 1); ones(5, 1)]');
figure
scatter(x(:,1), x(:,2), [], y, 'o');
hold on;
scatter(testDatax(:,1), testDatax(:,2), [], testDatay, 'filled');

%% 2)
% This code uses a synthetic dataset to test the performance of the
% training algorithm on a more complex scenario.
% The dataset contains points in R^2 clustered in 4 different classes, and
% has been generated using python script reported in the folder

% PRE-PROCESSING
clear all; close all; clc;
% synthetic dataset generated
load X_new.mat
load y_new.mat
y = Y;
% normalize dataset
% m = mean(X);
% s = max(X);
% for i = 1:2
%     X(:,i) = (X(:,i) + m(i))/s(i);   % normalize every dimension
% end

% parameters of the dataset
n = size(X, 1);
d = size(X, 2);
n_classes = 4;

figure
scatter(X(:, 1), X(:, 2), [], y);

% transform y to categorical (inefficient but needed)
y_cat = zeros(4, n);
for k = 1:n
    y_cat(y(k)+1,k) = 1;
end

% decide how to split the dataset
split_ratio = 0.3;
k = floor(split_ratio*n);

% split dataset into training and test
X_train = X(1:k,:);
X_test = X(k+1:end,:);
y_train = y(1:k);
y_test = y(k+1:end);
y_train_cat = y_cat(:,1:k);
y_test_cat = y_cat(:,k+1:end);

% define activation functions
sigma = @(t) 1./(1+exp(-t));
sigmaprime = @(t) sigma(t).*(1-sigma(t));

% define hyperparameters of the network ANY NETWORK SHAPE (this includes 
% input & output layers too)
shape = [d, 3, 3, n_classes];

niter = 3e5;    % max iterations
eta = 0.05;     % learning rate

% TRAINING
[costHistory, W, b] = GradientDescentClassification( ...
        X_train', y_train_cat, niter, sigma, sigmaprime, eta, shape);

% save the network
save NNparams.mat W b 

% plot training behaviour
figure
plot(linspace(0,niter,niter)', costHistory, '-')
fprintf('Cost Function: %f\n', costHistory(end));

filename = 'training_cost_function.png'; 
saveas(gcf, filename, 'png');

% TEST
load NNparams.mat W b

% predict the classes of the test points using the network
y_pred = PredictClasses(W, b, sigma, X_test');

% plot predictions
figure
scatter(X_train(:, 1), X_train(:, 2), [], y_train, 'o');
hold on;
scatter(X_test(:, 1), X_test(:, 2), [], y_pred, '*');
saveas(gcf, '4_classes.png', 'png');

% confusion matrix
figure
y_test = int32(y_test);
confusionchart(y_test,y_pred)
saveas(gcf, 'ConfusionMatrix', 'png');

% boundary exploration
load NNparams.mat W b
% generate new random data in the square [0,1]^2
n = 1000;
testDatax = -1 + 2*rand(n, 2);
testDatay = PredictClasses(W, b, sigma, testDatax');

figure
scatter(X_train(:,1), X_train(:,2), [], y_train, 'o');
hold on;
scatter(testDatax(:,1), testDatax(:,2), [], testDatay, 'filled');
saveas(gcf, 'SpaceExploration', 'png');