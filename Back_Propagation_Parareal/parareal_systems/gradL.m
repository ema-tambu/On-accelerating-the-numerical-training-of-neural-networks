function [L,dL] = gradL(y0)

% This function takes as input y0, which is a vector containing the Weight
% and the Biases at the previous step. It unfolds the weights and biases 
% and computes a forward-backward pass through the network, returning as 
% output the gradient of the Loss Function with respect to weight and 
% biases, all folded back in a vector

%  

% to unfold the vector of weights and biases we need to know the shape of
% the netwrok



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

% dL = delta{l}*(a{l})';
% dL = [dL; delta{l}];

% compute the total number of parameters of the network (bias included) and
% initialize dL = zeros(tot,1)

% n is the number of inputs, i.e. the dimension of the training dataset
for l = 1:L
    %1)  add the weights of the matrix W{l} to dL
        % the size of this matrix "temp" is independent of the number of input
        % (mxn)*(nxd)=(mxd) for all n
        temp = -delta{l}*a{l}'; % /size(x,2) if n>1 !!!
        for j = 1:size(temp,1)
            dL = [dL; temp(j,:)'];
        end
    % 2) add the weights of the bias b{l} to dL
        temp = -delta{l}; % use mean(delta{l},2) if n>1 !!!
        dL = [dL; temp];
end
    
% compute cost
% ok this is the value of the function L, not derived w.r.t anything, we
% just want to assure that its value is decreasing
cost = 0.5 * mean((a{end} - y).^2, 'all');
L = cost;
end