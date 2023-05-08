function [u_refined, t_refined] = Parareal_fun2(t0, tN, y0, f, dt, dT, maxIter)

% this function implements the parareal algorithm for a 1D ODE using forward euler. 
 
% t0:		initial time step 
% tN:		final time step 
% y0:		initial condition of the ODE 
% f:		ODE function 
% dt:		refined time step 
% dT:		coarse time step 
% t_coarse:		vector of the coarse time discretization 
% t_refined:	vector of the refined time discretization, one row for every sub interval 

% pre-allocate vectors and matrices
t_coarse = t0:dT:tN;
N_coarse = length(t_coarse);
N_refined = (mod(t_coarse(2)-t0, dt)==0)*(ceil((t_coarse(2)-t0)/dt)+1) + (mod(t_coarse(2)-t0, dt)>0)*(ceil((t_coarse(2)-t0)/dt));
[t_refined_base, t_coarse_offset] = meshgrid(0:dt:(N_refined-1)*dt, t_coarse(1:end-1));
t_refined = t_refined_base + t_coarse_offset;
u_refined = zeros(N_coarse-1, N_refined);

Uk = zeros(N_coarse, 1);

% step 0 - first approximation with coarse operator G
U = zeros(N_coarse, 1);
U(1) = y0;

for j = 1:N_coarse-1 
    U(j+1) = U(j) + dT*f(t_coarse(j),U(j)); 
end 

for i = 1:maxIter
    % parallel step
    parfor j = 1:N_coarse-1
        temp_u_refined = zeros(1, N_refined);
        temp_u_refined(1) = U(j);
        for k = 1:N_refined-1
            temp_u_refined(k+1) = temp_u_refined(k) + dt * f(t_refined(j, k), temp_u_refined(k));
        end
        u_refined(j, :) = temp_u_refined;
    end
    
    % prediction & correction 
    for j = 1:N_coarse-1 
        Uk(j+1) = U(j) + dT*f(t_coarse(j),U(j));
    end 
    for j = 1:N_coarse-1 
        U(j+1) = U(j) + dT*f(t_coarse(j), U(j)) + u_refined(j,end) - Uk(j+1);
    end 
    
    figure;
    hold on;
    for j = 1:N_coarse-1
        plot(t_refined(j,:), u_refined(j,:))
    end
    title(['Iteration k = ', num2str(i)]);

end
end