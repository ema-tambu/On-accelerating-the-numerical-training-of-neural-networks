clear all; close all; clc;

f = @(x) sin(10*x);

n_train = 50;
a_ = 0; b_ = 1;
x = sort(a_ + (b_ - a_).*rand(n_train, 1))';
y = f(x);

niter = 1e4;
shape = [1, 30, 30, 30, 30, 1];
sigma = @(t) tanh(t);
sigmaprime = @(t) 1 - tanh(t).^2;
eta = 0.005;

% [costHistory, W, b] = GradientDescentRegression(...
%     x, y, niter, shape, sigma, sigmaprime, eta);
load sin10xNNparams.mat

figure
plot(linspace(0,niter,niter)', costHistory, '-')
fprintf('Cost Function: %f\n', costHistory(end));

% Generate test data
n_test = 100;
x_test = linspace(0, 2, n_test);
y_test = f(x_test);

% Forward pass for test data
y_pred = PredictRegression(W, b, sigma, x_test);

% Plot true function, training data, and test predictions
figure;
plot(x_test, y_test, 'k-', 'LineWidth', 2);
hold on;
scatter(x, y, 'bo'); % Training data (blue dots)
scatter(x_test, y_pred, 'r.'); % Test predictions (red dots)
legend('True function', 'Training data', 'Test predictions', Location='best');
xlabel('x');
ylabel('y');
title('Neural Network Approximation of y = sin(10x)');
hold off;

% compute the error
errorL2 = norm(y_pred-y_test)
errorLp = norm(y_pred-y_test, 10)
errorInf = norm(y_pred-y_test, "inf")

%% regression of a function R2 -> R

clear all; close all; clc;

f = @(x1, x2) sin(10 * x1) + cos(10 * x2);
% Generate training data
n_train = 1000;
x1 = rand(1, n_train);
x2 = rand(1, n_train);
x = [x1; x2];
y = f(x1, x2);

% Generate test data
n_test = 30;
x1_test = linspace(0, 1, n_test);
x2_test = linspace(0, 1, n_test);
%  x_test = [x1_test; x2_test];
[X1_test, X2_test] = meshgrid(x1_test, x2_test);
X_test = [X1_test(:), X2_test(:)]';
Y_test = f(X1_test,X2_test);

niter = 5000;
shape = [2, 32, 64, 64, 32, 1];
% num_layers = length(shape);
sigma = @(t) tanh(t);
sigmaprime = @(t) 1 - tanh(t).^2;
eta = 0.01;

[costHistory, W, b] = GradientDescentRegression(...
        x, y, niter, shape, sigma, sigmaprime, eta);

figure
plot(linspace(0,niter,niter)', costHistory, '-')
fprintf('Cost Function: %f\n', costHistory(end));

y_pred = PredictRegression(W, b, sigma, X_test);
% Plot true function, training data, and test predictions
figure;
% subplot(1, 2, 1)
scatter3(x1, x2, y, 'bo'); % Training data (blue dots)
hold on;
% scatter3(X1_test(:), X2_test(:), Y_test(:), 'k.', 'LineWidth', 2); % True function (black dots)
surf(X1_test, X2_test, Y_test); % True function (black dots)

scatter3(X1_test(:), X2_test(:), y_pred, 'r.'); % Test predictions (red dots)
% y_pred_matrix = reshape(y_pred, [30, 30]);
% subplot(1, 2, 2)
% surf(X1_test, X2_test, y_pred_matrix); % Test predictions (red dots)

legend('Training data', 'True function', 'Test predictions');
xlabel('x1');
ylabel('x2');
zlabel('y');
title('Neural Network Approximation of y = sin(10 * x1) + cos(10 * x2)');
hold off;

errorL2 = norm(Y_test(:) - y_pred)