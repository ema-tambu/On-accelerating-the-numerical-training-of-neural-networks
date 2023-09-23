function [W, b] = TrainNetworkParareal( ...
        x, y, MaxIter, eta, sigma, sigmaprime, shape)
    % TrainNetworkParareal trains a given Feed Forward Neural Network using
    % the Parareal algorithm (parallel in time)
    %   x:              x-data of the training set
    %   y:              y-data of the training set
    %   MaxIter:        maximum number of iteration allowed
    %   sigma:          given activation function (it's going to be the 
    %                   same for all neurons)
    %   sigmaprime:     derivative of the activation function
    %   eta:            learning rate
    %   shape:          shape of the network

    L = length(shape);  % numer of layers

    % TODO: Here we could speedup things inizializing only the vector y0 
    % randomly

    % Initialize weights and biases randomly
    W = cell(1, L-1);
    b = cell(1, L-1);

    for l = 1:(L-1)
        W{l} = 0.5*randn(shape(l+1), shape(l));
        b{l} = 0.5*randn(shape(l+1), 1);
    end

    % TODO: add cost history to the output
    % costHistory = zeros(1, MaxIter);

    % PARAREAL
    
    % unfold matrices and biases into a vector that will be used by
    % Parareal [ {W,b} -> y0 ]
    
    % initialize y0
    tot_parameters = 0;
    for l = 1:L-1
        tot_parameters = tot_parameters + (shape(l+1)*shape(l)) + shape(l+1);
    end
    y0 = zeros(1,tot_parameters);

    % set initial conditions
    pointer_y0 = 1;
    for l = 1:(L-1)
        gap = shape(l) - 1;
        for j = 1:size(W{l},1)
            y0(1,pointer_y0:pointer_y0 + gap ) = W{l}(j,:)';
            pointer_y0 = pointer_y0 + gap + 1;
        end
        gap = shape(l+1)-1;
        y0(1,pointer_y0:pointer_y0 + gap) =  b{l};
        pointer_y0 = pointer_y0 + gap + 1;
    end
    
    % set final time T
    T = eta * MaxIter;

    % set number of parareal subintervals
    N = 4; 
    % set max number of parareal iterations
    parareal_max_iterations = N + 1;
    
    % 1) Parareal - PROBLEM IN HERE
    % [parareal_solution, iterations] = parareal_systems(T, N, y0, x, y, ...
    %     parareal_max_iterations, sigma, sigmaprime, shape);

    % 2) Non-parallel solver
    options = odeset('RelTol', 1e-3, 'AbsTol', 1e-3);
    [~, y45] = ode45(@(t, y_) gradL(y_, x, y, sigma, sigmaprime, shape), [0,T], y0, options);
    parareal_solution = y45(end,:);

    % finally rebuild the matrices [ y0 -> {W,b} ]
    pointer_y0 = 1;
    for l = 1:(L-1)
        % assign every row of the matrix
        gap = shape(l)-1;
        for i = 1:shape(l+1)
            W{l}(i,:) = parareal_solution(pointer_y0 : pointer_y0 + gap)';
            pointer_y0 = pointer_y0 + gap + 1;
        end
        % assign bias vector
        gap = shape(l+1)-1;
        b{l} = parareal_solution(pointer_y0:pointer_y0 + gap)';    % column vector
        pointer_y0 = pointer_y0 + gap + 1;
    end

    % costHistory(i) = cost; 
end
