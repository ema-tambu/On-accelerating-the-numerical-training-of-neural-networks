function [costHistory, W, b] = TrainNetworkParareal( ...
        x, y, MaxIter, sigma, sigmaprime, eta, shape)
    %GradientDescent Performs gradient descent on a given
    %Feed Forward Neural Network
    %   x:              x-data of the training set
    %   y:              y-data of the training set
    %   MaxIter:        maximum number of iteration allowed
    %   sigma:          given activation function (it's going to be the 
    %                   same for all neurons)
    %   sigmaprime:     derivative of the activation function
    %   eta:            learning rate
    %   shape:          shape of the network

    L = length(shape);  % numer of layers

    % Initialize weights and biases randomly
    W = cell(1, L-1);
    b = cell(1, L-1);

    for i = 1:(L-1)
        W{i} = 0.5*randn(shape(i+1), shape(i));
        b{i} = 0.5*randn(shape(i+1), 1);
    end

    costHistory = zeros(1, MaxIter);

    % training loop - BEGIN PARAREAL LOOP HERE
        f = @(cost,x) gradL(W{j-1},b{j-1}); % this is the function that
        % will be used as the right hand side of the ODE for the parareal.
        % Instead of calling this function multiple times through a 
        % function handle, we could save the output once and for all at
        % every iteration.

        % remember to save the value of the Loss Function at every step
        costHistory(i) = cost;

end
