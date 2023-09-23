function result = fine_solver(t0, y0, deltaT)
% disp(['t0 is ', num2str(t0)])
% disp(['Hi, Im doing my part from ', num2str(t0), ' to ', num2str(t0 + deltaT)])

% 1 - Use ODE45 for operator F() OR
% options = odeset('RelTol', 1e-3, 'AbsTol', 1e-3);
% [~, y] = ode45(@(t, y) ode_function(t, y), [t0, t0 + deltaT], y0, options);
% result = y(end,:);

% 2 - OR a fine stepped Fowrward Euler

% % n = 9000;    % reducing step for the fine solver
% % dt = deltaT/n;
% 
dt = 0.05;  % this is actually the old eta
n = deltaT/dt;
for j =1:n
    y0 = y0 + dt * ode_function(t0, y0)';
    t0 = t0 + dt;
end
result = y0;

% disp('Im done, returning')

end
