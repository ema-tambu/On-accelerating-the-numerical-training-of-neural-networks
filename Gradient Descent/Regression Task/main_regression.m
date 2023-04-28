clear all; close all; clc;

f = @(x) sin(10*x);

num_samples = 50;
a_ = 0; b_ = 1;
x = sort(a_ + (b_ - a_).*rand(num_samples, 1))';
y = f(x);

max_iterations = 500;
network_shape = [1, 30, 30, 30, 30, 1];
activation_function = @(t) tanh(t);

[W, b, costHistory] = GradientDescentRegression(x, y, max_iterations, network_shape);

figure
plot(linspace(0,max_iterations,max_iterations)', costHistory, '-')
fprintf('Cost Function: %f\n', costHistory(end));

% Generate test data
num_test_samples = 100;
x_test = linspace(0, 1, num_test_samples);
y_test = f(x_test);

% Forward pass for test data
num_layers = length(network_shape);
A_test = cell(1, num_layers);
A_test{1} = x_test;
for i = 1:num_layers - 1
    A_test{i+1} = activation_function(W{i} * A_test{i} + b{i});
end
y_pred = A_test{end};

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

%% regression of a function R2 -> R

clear all; close all; clc;
% Generate training data
num_samples = 1000;
x1 = rand(1, num_samples);
x2 = rand(1, num_samples);
x = [x1; x2];
y = sin(10 * x1) + cos(10 * x2);

% Generate test data
num_test_samples = 30;
x1_test = linspace(0, 1, num_test_samples);
x2_test = linspace(0, 1, num_test_samples);
%  x_test = [x1_test; x2_test];
[X1_test, X2_test] = meshgrid(x1_test, x2_test);
X_test = [X1_test(:), X2_test(:)]';
Y_test = sin(10 * X1_test) + cos(10 * X2_test);

max_iterations = 5000;
network_shape = [2, 32, 32, 32, 32, 1];
num_layers = length(network_shape);
activation_function = @(t) tanh(t);

[W, b, costHistory] = GradientDescentRegressionR2(x, y, max_iterations, network_shape);
%% 
% Forward pass for test data
A_test = cell(1, num_layers);
A_test{1} = X_test;
for i = 1:num_layers - 1
    A_test{i+1} = activation_function(W{i} * A_test{i} + b{i});
end
y_pred = A_test{end};

% Plot true function, training data, and test predictions
figure;
subplot(1, 2, 1)
scatter3(x1, x2, y, 'bo'); % Training data (blue dots)
hold on;
% scatter3(X1_test(:), X2_test(:), Y_test(:), 'k.', 'LineWidth', 2); % True function (black dots)
% scatter3(X1_test(:), X2_test(:), y_pred, 'r.'); % Test predictions (red dots)
surf(X1_test, X2_test, Y_test); % True function (black dots)
y_pred_matrix = reshape(y_pred, [30, 30]);
subplot(1, 2, 2)
surf(X1_test, X2_test, y_pred_matrix); % Test predictions (red dots)

% legend('Training data', 'True function', 'Test predictions');
xlabel('x1');
ylabel('x2');
zlabel('y');
title('Neural Network Approximation of y = sin(10 * x1) + cos(10 * x2)');
hold off;