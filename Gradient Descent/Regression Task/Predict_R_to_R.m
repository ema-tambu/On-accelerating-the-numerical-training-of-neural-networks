function [Predy] = Predict_R_to_R(W, b, sigma, Datax)
 % length of the network
    L = numel(W);
    % length of the test dataset
    N = size(Datax, 2); % R^n
    % pre-allocate output
    Predy = zeros(1,N);

    for i=1:N
        x = Datax(:,i);
        for l = 1:L
            x = sigma(W{l}*x+b{l});
        end
        Predy(i) = x;
    end
end