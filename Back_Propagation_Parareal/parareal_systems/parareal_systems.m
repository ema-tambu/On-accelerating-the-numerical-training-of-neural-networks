function [U, k] = parareal_systems(T, N, y0, max_iterations, sigma, sigmaprime, shape)
% sigma, sigmaprime, shape are ghost parameter that are being carried for
% calls of nested functions who need them
deltaT = T / N;

dim = length(y0);

U_tilde = zeros(N + 1, dim);
U_tilde(1,:) = y0;

U_hat = zeros(N + 1, dim);
U_hat(1,:) = y0;

Utemp = zeros(N + 1, dim);

% First iteration with coarse solver
for j =1:N
    U_tilde(j + 1,:) = coarse_solver((j-1)*deltaT, U_tilde(j,:), deltaT, sigma, sigmaprime, shape);
end

U = U_tilde;

% Parareal iterations
k = 1;
while k < max_iterations

    % Parallel fine solver step
    parfor j =1:N
        U_hat(j,:) = fine_solver((j-1)*deltaT, U_tilde(j,:), deltaT, sigma, sigmaprime, shape);
    end

    % Update coarse solution
    for j=1:N
        Utemp(j+1,:) = coarse_solver((j-1)*deltaT, U(j,:), deltaT, sigma, sigmaprime, shape);
    end

    % Sequential update step
    for j =1:N
        U(j+1,:) = coarse_solver((j-1)*deltaT, U(j,:), deltaT, sigma, sigmaprime, shape);
        U(j+1,:) = U(j+1,:) + U_hat(j,:) - Utemp(j+1,:);
    end
    
    % TODO: Check for convergence
    % if norm(U[1:] - U_tilde[1:]) < tolerance:
    %    break

    U_tilde = U;

    k = k + 1;

end
end