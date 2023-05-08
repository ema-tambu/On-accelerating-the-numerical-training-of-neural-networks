clear all
close all
clc

f = @(t,y) sin(t).*y + t;
%f = @(t,y) cos(t).*y - t;
t0 = 0; tN = 40;
y0 = 0;
dt = 0.0002;
dT = 1;
k_max = 4;

tic
% [u,t] = fwd_Euler_parareal(t0, tN, y0, f, dt, dT, k_max);
[u,t] = fwdEuler_Parareal_2(t0, tN, y0, f, dt, dT, k_max);
time_parareal = toc

tic
[t_fwd, u_fwd] = fwd_Euler(t0,tN,y0,dt,f);
time_fwdEuler = toc

% Plot of the last iteration of Parareal
% close all
figure
plot(t_fwd,u_fwd,'-','Linewidth',2, 'Color', 'r') % fine
hold on
for n = 1:(tN - t0)/dT
    plot(t(n,1),u(n,1),'o-','Linewidth',2, 'Color', 'b') % first point of the subinterval
    plot(t(n,2:end),u(n,2:end),'--','Linewidth',2, 'Color', 'b') % fine by parareal
end
legend('Euler','Parareal')