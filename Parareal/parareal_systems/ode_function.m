function result = ode_function(t,y)

% result = [2*y(1)-y(1).*y(1)+0.*t; - 3*y(2)+cos(y(2))+0.*t];

% result = [ sin(t).*y(1)+t; cos(t).*y(2)-t ];
% result = [
%     -3*y(1) + 0.*t + 0.2.*y(2).*log(-t+1);
%     sin(t).*cos(y(2)) + exp(-t) - y(1)
%     ];

% Lotka Volterra System two species
a = 0.1; b = 0.02; g = 0.4; d = 0.02;
result = [
    y(1).*(a-b*y(2));
    y(2).*(d*y(1)-g)
    ];

% Lotka Volterra System three species
% a = 1; b = 1; c = 2; d = 2; e = 2; f = 1.1; g = 1;
% result = [
%     a*y(1) - b*y(1)*y(2); 
%     -c*y(2) + d*y(1)*y(2) - e*y(2)*y(3); 
%     -f*y(3) + g*y(2)*y(3)
%     ];

end