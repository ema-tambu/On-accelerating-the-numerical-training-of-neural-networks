function result = fine_solver(t0, y0, deltaT)

% 1 - Use ODE45 for operator F() OR

options = odeset('RelTol', 1e-6, 'AbsTol', 1e-6);
[~, y] = ode45(@(t, y) ode_function(t, y), [t0, t0 + deltaT], y0, options);
result = y(end);

% 2 - OR a fine stepped Fowrward Euler

% n = 150;    % reducing step for the fine solver
% dt = deltaT/n;
% for j =1:n
%     y0 = y0 + dt * ode_function(t0, y0);
%     t0 = t0 + dt;
% end
% result = y0;

end