function [W, b, costHistory] = GradientDescentRegression(x, y, max_iterations, network_shape)

    % Validate network shape
    if network_shape(1) ~= 1 || network_shape(end) ~= 1
        error('Network shape must begin and end with 1');
    end
    
    % Number of layers
    num_layers = length(network_shape);
    
    % Initialize weights and biases
    W = cell(1, num_layers - 1);
    b = cell(1, num_layers - 1);
    
    for i = 1:num_layers - 1
        W{i} = randn(network_shape(i+1), network_shape(i));
        b{i} = randn(network_shape(i+1), 1);
    end
    
    % Learning rate and cost history
    learning_rate = 0.01;
    costHistory = zeros(1, max_iterations);
    
    % Training loop
    for iter = 1:max_iterations
        % Forward pass
        A = cell(1, num_layers);
        Z = cell(1, num_layers - 1);
        A{1} = x;
        for i = 1:num_layers - 1
            Z{i} = W{i} * A{i} + b{i};
            A{i+1} = activation_function(Z{i});
        end
    
        % Compute cost
        cost = 0.5 * mean((A{end} - y).^2);
        costHistory(iter) = cost;
    
        % Backward pass
        dZ = cell(1, num_layers - 1);
        dW = cell(1, num_layers - 1);
        db = cell(1, num_layers - 1);
        dZ{end} = (A{end} - y) .* activation_derivative(Z{end});
    
        for i = num_layers - 1:-1:1
            dW{i} = (dZ{i} * A{i}') / size(x, 2);
            db{i} = mean(dZ{i}, 2);
            if i > 1
                dZ{i-1} = (W{i}' * dZ{i}) .* activation_derivative(Z{i-1});
            end
        end
    
        % Update weights and biases
        for i = 1:num_layers - 1
            W{i} = W{i} - learning_rate * dW{i};
            b{i} = b{i} - learning_rate * db{i};
        end
    end

end % end function

function y = activation_function(x)
y = tanh(x);
end

function y = activation_derivative(x)
y = 1 - tanh(x).^2;
end
