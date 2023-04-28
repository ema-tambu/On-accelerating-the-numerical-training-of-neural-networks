% this codes trains a NN to learn functions from R to R

clear all; close all; clc;

f = @(x) sin(10*x); % Replace this with your desired function

% take some data for y = sin(x) in the interval [0,1]
N_train = 50;
a_ = 0; b_ = 1;
x = sort(a_ + (b_ - a_).*rand(N_train, 1))';
y = f(x);

% scatter(x, y)

% define ReLU activation functions
% sigma = @(t) t.*(t>0);
% sigmaprime = @(t) (t>0);

% leaky ReLU
% sigma = @(t) max(0.01*t, t);
% sigmaprime = @(t) (t > 0) + 0.01*(t <= 0);

% sigmoid isn't appropriate for regression task
% sigma = @(t) 1./(1+exp(-t));
% sigmaprime = @(t) sigma(t).*(1-sigma(t));

sigma = @(t) tanh(t);
sigmaprime = @(t) 1- tanh(t).^2;

shape = [1, 32, 32, 32, 1];
niter = 500;
eta = 0.01;

[costHistory, W, b] = GradientDescentClassification( ...
        x, y, niter, sigma, sigmaprime, eta, shape);

figure
plot(linspace(0,niter,niter)', costHistory, '-')
fprintf('Cost Function: %f\n', costHistory(end));

N_test = 10;
x_test = sort(a_ + (b_ - a_).*rand(N_test, 1))';
y_test = f(x_test);
y_pred = Predict_R_to_R(W, b, sigma, x_test)';


figure
plot(x, y, 'o-')
hold on
plot(x_test, y_pred, 'o')

xx = linspace(a_, b_, 300);
yy = f(xx);
plot(xx, yy)
legend('train', 'test', 'exact')