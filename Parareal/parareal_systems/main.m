clear all; close all; clc;

T = 20000;
N = 6;
y0 = [10;10];
% y0 = [1; 1; 1]; % Lotka Volterra
time_points = linspace(0, T, N + 1);

% maxit = N + 1;
% tic
% [parareal_solution, iterations] = parareal_systems(T, N, y0, maxit);
% time_parareal = toc

disp('Parareal solution:')
% disp(parareal_solution)
% disp(size(parareal_solution))

% disp('Number of iterations: ')
% disp(iterations)

tic
[time_reference, reference_sol] = reference_solution(T, y0);
time_ode45 = toc

disp('Reference solution: ')
% disp(reference_sol')
disp(size(reference_sol))

% disp('end_parareal:')
% disp(parareal_solution(end,:))
disp('end_reference:')
disp(reference_sol(end,:))

% disp('error reference - parareal')
% disp(norm(reference_sol(end,:) - parareal_solution(end,:), 2))

% plot_solution(time_points, parareal_solution, time_reference, reference_sol)

%%
dim = size(reference_sol, 2);
figure
hold on;
for i=1:dim
    plot(time_reference(end-100:end), reference_sol(end-100:end,i), '.--', 'DisplayName', 'Reference Solution');
end
hold off;
xlabel('Time');
ylabel('y(t)');
legend();
grid on;

