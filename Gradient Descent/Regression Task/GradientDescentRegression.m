function [costHistory, W, b] = GradientDescentRegression(...
    x, y, MaxIter, shape, sigma, sigmaprime, eta)

    % Validate network shape
    if shape(end) ~= 1
        error('Network shape provided is not for regression');
    end
    
    % Number of layers
    L = length(shape);
    
    % Initialize weights and biases
    W = cell(1, L-1);
    b = cell(1, L-1);
    
    z = cell(1,L-1);
    D = cell(1,L-1);
    delta = cell(1, L-1);
    a = cell(1,L);

    for l = 1:L-1
        W{l} = randn(shape(l+1), shape(l));
        b{l} = randn(shape(l+1), 1);
    end
    costHistory = zeros(1, MaxIter);
    
    % Training loop
    temp = 0.1;    % just a handle variable to print the progress
    for i = 1:MaxIter
        % Forward pass
%         A = cell(1, L);
%         Z = cell(1, L - 1);
        a{1} = x;
        for l = 1:L - 1
            z{l} = W{l}*a{l}+b{l};
            a{l+1} = sigma(z{l});
%             D{l} = diag(sigmaprime(z{l}));
        end
    
        % Compute cost
        cost = 0.5 * mean((a{end} - y).^2);
        costHistory(i) = cost;
    
        % Backward pass
%         dZ = cell(1, L - 1);
%         dW = cell(1, L - 1);
%         db = cell(1, L - 1);
%         dZ{end} = (A{end} - y) .* activation_derivative(Z{end});
        delta{L-1} = (a{L}-y).*sigmaprime(z{L-1});
        for l = L-1:-1:2
            delta{l-1} = sigmaprime(z{l-1}).*((W{l})'*delta{l});
        end
%         for l = L - 1:-1:1
%             dW{l} = (dZ{l} * A{l}') / size(x, 2);
%             db{l} = mean(dZ{l}, 2);
%             if l > 1
%                 dZ{l-1} = (W{l}' * dZ{l}) .* activation_derivative(Z{l-1});
%             end
%         end
    
        % Update weights and biases
        for l = 1:L - 1
%             W{l} = W{l} - eta * dW{l};
%             b{l} = b{l} - eta * db{l};
            % here I no longer have the stochastic version
            W{l} = W{l} - eta*((delta{l}*a{l}')/size(x,2));
            b{l} = b{l} - eta*mean(delta{l}, 2);
            
        end
        
        % Print progress on the screen (10%, 20%, ...)
        if i == floor(MaxIter*temp)
            fprintf('%.1f%%\n', temp*100);
            temp = temp + 0.1;
        end
    end

end % end function

% function y = activation_function(x)
% y = tanh(x);
% end
% 
% function y = activation_derivative(x)
% y = 1 - tanh(x).^2;
% end
