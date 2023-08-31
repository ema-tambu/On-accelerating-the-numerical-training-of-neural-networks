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

    for l = 1:(L-1)
        W{l} = 0.5*randn(shape(l+1), shape(l));
        b{l} = 0.5*randn(shape(l+1), 1);
    end

    costHistory = zeros(1, MaxIter);

    % BEGIN PARAREAL HERE
    
    % prepare initial conditions (unfold matices and biases)
    for l = 1:(L-1)
        for j = 1:size(W{l},1)
            y0 = [y0; W(j,:)'];
        end
        y0 = [y0; b{l}];
    end
    
    T = 1; %the absolute time is not important, what's important is how fast you grow in such time
    N = 4; 

    % TODO: c'è da passare in qualche modo sigma e sigmaprime a gradL !!

    [parareal_solution, iterations] = parareal_systems(T, N, y0, MaxIter);

    % remember to save the value of the Loss Function at every step
    costHistory(i) = cost;

end
