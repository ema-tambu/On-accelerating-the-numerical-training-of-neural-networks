function [L,dL] = gradL(y0, sigma, sigmaprime, shape)

% This function takes as input y0, (no specific time is needed!!) which is a vector containing the Weight
% and the Biases at the previous step. It unfolds the weights and biases 
% and computes a forward-backward pass through the network, returning as 
% output the gradient of the Loss Function with respect to weight and 
% biases, all folded back in a vector

% to unfold the vector of weights and biases we need to know the shape of
% the netwrok

% Inizialize memory
W = cell(1, L-1);
b = cell(1, L-1);

parfor l = 1:(L-1)
    W{l} = 0.5*randn(shape(l+1), shape(l));
    b{l} = 0.5*randn(shape(l+1), 1);
end

% compose the set of matrices and weights from the vector y0
pointer_y0 = 1;
for l = 1:(L-1)
    % assign every row of the matrix
    gap = shape(l)-1;
    for i = 1:shape(l+1)
        W{l}(i,:) = y0(pointer_y0 : pointer_y0 + gap)';
        pointer_y0 = pointer_y0 + gap + 1;
    end
    % assign bias vector
    gap = shape(l+1)-1;
    b{l} = y0(pointer_y0:pointer_y0 + gap);
    pointer_y0 = pointer_y0 + gap + 1;
end

% compute the derivative of the Loss function w.r.t every wheight
% initialize structures
z = cell(1,L-1);
delta = cell(1, L-1);
a = cell(1,L);

% forward pass
a{1} = x;
for l = 1:(L-1)
    z{l} = W{l}*a{l}+b{l};
    a{l+1} = sigma(z{l});
end

% backward pass
delta{L-1} = (a{L}-y).*sigmaprime(z{L-1});
for l = (L-1):-1:2
    delta{l-1} = sigmaprime(z{l-1}).*(W{l}'*delta{l});
end


% Here I need to return the dL in a vector
for l = 1:(L-1)
    W{l} = delta{l}*(a{l})'; % (delta{l}*(a{l})')/size(x,2); % this is needed for full gradient descent, not stochastic
    b{l} = delta{l}; %mean(delta{l},2) % this is needed for full gradient descent, not stochastic
end

% now transform back again the matrices into a vector
% prepare initial conditions (unfold matrices and biases)


% recycle memory of y0 that now becomes dL
pointer_y0 = 1;
for l = 1:(L-1)
    gap = shape(l) - 1;
    for j = 1:size(W{l},1)
        y0 = [y0; W(j,:)'];
        y0(pointer_y0:pointer_y0 + gap ) = W(j,:)';
        pointer_y0 = pointer_y0 + gap + 1;
    end
    gap = shape(l+1)-1;
    y0(pointer_y0:pointter_y0 + gap) =  b{l};
    pointer_y0 = pointer_y0 + gap + 1;
end
dL = y0;    % I hope this is a reference assignement and not a copy constructor

% compute cost
% ok this is the value of the function L, not derived w.r.t anything, we
% just want to assure that its value is decreasing
cost = 0.5 * mean((a{end} - y).^2, 'all');
L = cost;

end