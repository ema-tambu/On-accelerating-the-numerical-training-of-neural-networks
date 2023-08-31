function plot_solution(time_points, parareal_solution, time_reference, reference_solution)

plot(time_points, parareal_solution, 'o', 'MarkerSize', 10, 'DisplayName', 'Parareal Solution');
hold on;
plot(time_reference, reference_solution, '.--', 'DisplayName', 'Reference Solution');

hold off;
xlabel('Time');
ylabel('y(t)');
legend();
grid on;
end