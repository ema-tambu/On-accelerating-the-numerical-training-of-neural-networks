function [U_coarse, k] = parareal_systems(T, N_coarse, N_fine, y0, x, y, sigma, sigmaprime, shape)
% sigma, sigmaprime, shape are ghost parameter that are being carried for
% calls of nested functions who need them
dT = T / N_coarse;
dt = T/ (N_coarse * N_fine);

dim = length(y0);

% initialize buffers
U_coarse = zeros(N_coarse + 1, dim);
U_coarse_temp = zeros(N_coarse + 1, dim);
U_fine = zeros(N_coarse, dim);

U_coarse(1,:) = y0;
U_coarse_temp(1,:) = y0;

% zeroth iteration
for i =1:N_coarse
    U_coarse_temp(i + 1,:) = coarse_solver((i-1)*dT, U_coarse_temp(i,:), ...
        dT, x, y, sigma, sigmaprime, shape);
end

% parareal loop
for k = 1:N_coarse

    % parallel for (fine solver)
    parfor i = 1:N_coarse
        U_fine(i,:) = fine_solver((i-1)*dT, U_coarse(i,:), dt, dT, x, y, sigma, sigmaprime, shape);
    end

    % predict - correct
    % for i = 1:N_coarse
    for i = k:N_coarse  % think we can optimize like this
        bff1 = coarse_solver((i-1)*dT, U_coarse(i,:), dT, x, y, sigma, sigmaprime, shape);
        bff2 = coarse_solver((i-1)*dT, U_coarse_temp(i,:), dT, x, y, sigma, sigmaprime, shape);
        U_coarse(i+1,:) = bff1 + U_fine(i,:) - bff2;
    end
    
    % check for convergence
    incr = norm(U_coarse - U_coarse_temp, 2);
    if (incr < 1e-3)
        disp(['Parareal converged at iteration ' num2str(k) '/' num2str(N_coarse)])
        break
    end
    U_coarse_temp = U_coarse;

    disp(['iteration ' num2str(k) '/' num2str(N_coarse)])
end

end