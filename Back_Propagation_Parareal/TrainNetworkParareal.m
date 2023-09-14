function [___decide_what_output_here____] = TrainNetworkParareal( ...
        x, y, MaxIter, sigma, sigmaprime, shape)
    %GradientDescent Performs gradient descent on a given
    %Feed Forward Neural Network
    %   x:              x-data of the training set
    %   y:              y-data of the training set
    %   MaxIter:        maximum number of iteration allowed
    %   sigma:          given activation function (it's going to be the 
    %                   same for all neurons)
    %   sigmaprime:     derivative of the activation function
                %   eta:            learning rate (eta is NOT needed anymore)
    %   shape:          shape of the network

    L = length(shape);  % numer of layers

    % Here we can speedup things and inizialize only the vector y0 randomly

    % Initialize weights and biases randomly
    W = cell(1, L-1);
    b = cell(1, L-1);

    for l = 1:(L-1)
        W{l} = 0.5*randn(shape(l+1), shape(l));
        b{l} = 0.5*randn(shape(l+1), 1);
    end

    costHistory = zeros(1, MaxIter);

    % BEGIN PARAREAL HERE
    % initialize y0 by counting the total number of parameters in the
    % network
    tot_parameters = 0;
    for l = 1:L-1
        tot_parameters = tot_parameters + (shape(l+1)*shape(l)) + shape(l+1);
    end
    y0 = zeros(1,tot_parameters);
    % prepare initial conditions (unroll matrices and biases into a vector)
    pointer_y0 = 1;
    for l = 1:(L-1)
        gap = shape(l) - 1;
        for j = 1:size(W{l},1)
            y0(pointer_y0:pointer_y0 + gap ) = W(j,:)';
            pointer_y0 = pointer_y0 + gap + 1;
        end
        gap = shape(l+1)-1;
        y0(pointer_y0:pointter_y0 + gap) =  b{l};
        pointer_y0 = pointer_y0 + gap + 1;
    end
    
    T = 1; % What is important, I think, it's the ratio between T and dT, which should be al large as possible. But tha absolute final time step is not important.
    N = 4; 

    % Here we need to pass sigma, sigmaprime and shape to the
    % parareal_system function, since these parameters are needed in the 
    % nested function gradL. Unfortunately I don't know any other solution
    % if we want to split the code in multiple files (possible solution:
    % switch to OOP)
    [parareal_solution, iterations] = parareal_systems(T, N, y0, MaxIter, sigma, sigmaprime, shape);

    % remember to save the value of the Loss Function at every step
    % costHistory(i) = cost; % aspe', in un secondo momento

end
