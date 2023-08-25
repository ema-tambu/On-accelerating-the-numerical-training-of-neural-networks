function dL = gradL(W, b)
% This function takes as input the Weight and the Biases at the
% previous step and computes a forward-backward pass through the
% network, returning as output the gradient of the Loss Function with
% respect to weight and biases

%   W:              weights of the network at the previous time step
%   b:              biases of the network at the previous time step
%   dL:             gradient of the Loss Function w.r.t weights and biases

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

dL = delta{l}*(a{l})';
dL = [dL; delta{l}];
end