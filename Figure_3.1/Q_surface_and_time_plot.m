% Computes the Q surface and Q vs time for a linear policy (or a rough
% approximation to a linear policy) comntrolling a linear, purely inertial,
% 1-DOF system --- a 1-kg mass on a horizontal frictionless rail --- 
% with a quadratic or tanh reward and a discount factor and a time step of 1.

clear variables;
clc;
clf;

% Set task parameters
dur = 5000;  % duration of each rollout in ms
m = 1;  % mass of cart in kg
alpha_1 = -5e-4; alpha_0 = (5e-3)*alpha_1;  % coeff's of desired dynamics: accel = alpha_1*vel + alpha_0*pos
gamma = 0.99;  % "discount" factor
grid_width = 51; 
V = zeros(grid_width, grid_width);
M = 3;  % initial position range is [-M, M]; vel range 1e-3*[-M, M]
fit_mu = 1;
fit_Q = 1;
rng(21);  % 21

% Set up network preprocessing and data recording
path_scale = (grid_width - 1)/(2*M); path_offset = (grid_width + 1)/2;
DATA = zeros(2, 1 + dur);  % allocate memory to record one rollout trajectory

if fit_mu == 1

% Initialize the RLS Gaussian policy
nx = 2;  % # elements in input x 
dom = M;  % radius of domain, i.e. pos and scaled vel
%nf = 500; nidvarg = -0.1/dom*dom;  % good fit; nidvarg is neg inv of double the variance of the Gaussians
nf = 30;  nidvarg = -10/dom*dom;  % poor fit  50 10
%nf = 10;  nidvarg = -0.1/dom*dom;  % poor fit
CEN = zeros(nx, nf);  % allocate memory for centres matrix
w = zeros(1, nf);  % initial weights
P = 1e10*eye(nf);  % soft initialization of RLS
y_sum = 0; y_sqr_sum = 0;

% Choose features, i.e. Gaussian centres in rescaled state space (with vel mult'd by 1e3)
for f = 1:nf
   i = floor(grid_width*rand + 1); j = floor(grid_width*rand + 1);
   CEN(:, f) = [(i - path_offset)/path_scale; (j - path_offset)/path_scale];
end

% Train policy mu
for i = 1:grid_width, for j = 1:grid_width
  q = (i - path_offset)/path_scale; q_vel = 1e-3*(j - path_offset)/path_scale;  % *****
  a = m*(alpha_1*q_vel + alpha_0*(q - 0.3));  % desired output
  x = [q; 1e3*q_vel];  % scale vel for input to policy *****
  D = CEN - x; phi = exp(nidvarg*sum(D.*D)');  % feature vector
  y_ = w*phi; e = y_ - a;  % estimate of desired output; model error
  Pphi = P*phi; k = Pphi'/(1 + phi'*Pphi); P = P - Pphi*k; w = w - e*k;  % RLS
end, end

end  % if fit_mu

% Run many rollouts from a grid of initial states
for i = 1:grid_width, for j = 1:grid_width

% Initialize rollout
t = 0; 
q = -M + (i - 1)*2*M/(grid_width - 1); 
q_vel = 1e-3*(-M + (j - 1)*2*M/(grid_width - 1));  % *****
gamma_t = 1;  
G = 0;  % initial return

% Save one of the rollout trajectories for plotting
i_path = 42; j_path = 1 + round((10/50)*(grid_width - 1));  % record 42/51, 10/51
if (i == i_path) && (j == j_path), i_gp = 0; end  % i_gp = index of graph pts;

% Run rollout
for t = 0:dur

  if fit_mu == 0
    a = m*(alpha_1*q_vel + alpha_0*(q - 0.3));  % action = force in newtons
  else
    D = CEN - [q; 1e3*q_vel];  % scale vel for input to mu *****
    phi = exp(nidvarg*sum(D.*D)'); a = w*phi;
  end
  r = -0.3*(q*q*1e-3 + 10*q_vel*q_vel*1e3);  % quadratic reward *****
  %r = -0.035*tanh(10*q*q);  % nonquadratic reward
  G = G + gamma_t*r;  % return

  % Record path
  if (i == i_path) && (j == j_path)
    i_gp = i_gp + 1; DATA(:, i_gp) = [q; q_vel];
  end

  % Update dynamics and discount
  q_acc = a/m; q = q + q_vel; q_vel = q_vel + q_acc;
  gamma_t = gamma*gamma_t;

end  % t
V(j, i) = G;  % pos varies along rows, vel along columns

end, end
V_max = max(max(V));

if fit_Q == 1

% Initialize the RLS Gaussian Q-estimator
nx = 2;  % # elements in input x
n_ex = grid_width*grid_width;  % # examples
dom = M;  % radius of q domain
nf = 500; nidvarg = -0.1/dom*dom;  % nidvarg is neg inv of double the variance of the Gaussians
%nf = 500; nidvarg = -3/dom*dom;  % good fit when reward is tanh
%nf = 50;  nidvarg = -1/dom*dom;  % poor fit
CEN = zeros(nx, nf);  % allocate memory for centres matrix
w = zeros(1, nf);  % initial weights
P = 1e10*eye(nf);  % soft initialization
y_sum = 0; y_sqr_sum = 0;

% Choose features
for f = 1:nf
   i = floor(grid_width*rand + 1); j = floor(grid_width*rand + 1);
   CEN(:, f) = [(i - path_offset)/path_scale; (j - path_offset)/path_scale];
end

% Train Q-estimator
for i = 1:grid_width, for j = 1:grid_width
  x = [(i - path_offset)/path_scale; (j - path_offset)/path_scale];  % scale vel input to critic *****
  y = V(j, i);  % desired output, arranged so pos varies along rows
  D = CEN - x; phi = exp(nidvarg*sum(D.*D)');  % feature vector
  y_ = w*phi; e = y_ - y;  % estimate of y; model error
  Pphi = P*phi; k = Pphi'/(1 + phi'*Pphi); P = P - Pphi*k; w = w - e*k;  % RLS
end, end

% Assess Q-estimator
e_sqr_sum = 0;
for i = 1:grid_width, for j = 1:grid_width
  x = [(i - path_offset)/path_scale; (j - path_offset)/path_scale];
  y = V(j, i);  % desired output
  y_sum = y_sum + y; y_sqr_sum = y_sqr_sum + y*y;  % statistics of y
  D = CEN - x; phi = exp(nidvarg*sum(D.*D)');  % feature vector
  y_ = w*phi; e = y_ - y;  % estimate of y; model error
  e_sqr_sum = e_sqr_sum + e*e;
  V_(j, i) = y_;
end, end
y_avg = y_sum/n_ex; y_var = y_sqr_sum/n_ex - y_avg*y_avg;
NMSE = e_sqr_sum/(grid_width*grid_width*y_var)

end  % if fit_Q

% Plot

% Set size, shape, and location of picture
figure(1)
screen = get(0, 'ScreenSize');  % [left bottom width height] in pixels, e.g. [1 1 1920 1080]
wide = round(0.5*screen(3));
tall = round(0.8*screen(4));
h_margin = round(0.25*screen(3));
v_margin = round(0.075*screen(4));
set(figure(1), 'OuterPosition', [h_margin, screen(4) - tall - v_margin, wide, tall]);  % [left bottom width height]
statespace_height = 1.999;  % place the state-space rectangle this much higher than the Q-surface

% Set axes of left-panel spatial plots
subplot(1, 2, 1);
view(90, 30);
set(gca, 'Position', [0.1, 0.4, 0.38, 0.4]);

% Plot one rollout-trajectory in state space
DATA_scaled = path_scale*([1; 1e3].*DATA(:, 1:i_gp)) + path_offset;  % *****
O = ones(2, i_gp);
hold on
plot3(DATA_scaled(1, :), DATA_scaled(2, :), statespace_height*O, 'b');

% Plot the true or estimated Q-surface
if fit_Q == 0
  s = surf(V);  % plot the true on-policy action-value
else
  s = surf(V_);  % plot the estimate
end  
s.EdgeColor = 'none';
s.FaceColor = [1.0, 1.0, 0.99];  %'interp'
%alpha 0.25
lighting gouraud;
camlight(0, 90);
view(15, 25);
xlim([1, grid_width]);
ylim([1, grid_width]);
zlim([-4, statespace_height]);  % *****
xticks([1 + (grid_width - 1)/6, 1 + (grid_width - 1)/2, 1 + (grid_width - 1)*5/6])
xticklabels({'-2', '0', '2'})
yticks([1 + (grid_width - 1)/6, 1 + (grid_width - 1)/2, 1 + (grid_width - 1)*5/6])
yticklabels({'-2', '0', '2'})
zticks([-4, -2, 0])
zticklabels({'-4', '-2', '0'})
xlabel('pos')
ylabel('vel')
zlabel('Q');

% Rotate and reposition x- and y-axis labels
xh = get(gca,'XLabel'); % handle of the x label
set(xh, 'Units', 'Normalized')
pos = get(xh, 'Position');
set(xh, 'Position', pos + [-0.04, -0.01, 0], 'Rotation', -10)
yh = get(gca,'YLabel'); % handle of the y label
set(yh, 'Units', 'Normalized')
pos = get(yh, 'Position');
set(yh, 'Position', pos + [-0.01, 0.05, 0.0], 'Rotation', 55)

% Show state space as a horizontal blue rectangle 
Statespace = surf(statespace_height*ones(grid_width, grid_width));
Statespace.EdgeColor = 'none';
Statespace.FaceColor = [0.0 0.5 1.0];
Statespace.FaceAlpha = 0.2;

if fit_Q == 1

% Plot path on estimated Q-surface
for p = 1:i_gp
  x = DATA(:, p);
  x = [x(1); 1e3*x(2)];
  D = CEN - x; phi = exp(nidvarg*sum(D.*D)');  % feature vector
  y(p) = w*phi;
  plot3(DATA_scaled(1, p), DATA_scaled(2, p), y(p), '.', 'Color', [0.09 0.09 0.07], 'MarkerSize', 1, 'LineWidth', 1.5);
end

% Plot Q-estimate vs time
subplot(1, 2, 2);
set(gca, 'Position', [0.57, 0.4357, 0.38, 0.3286]);
plot(y, 'k');
xlim([0, dur]);
ylim([-4, 1]);
set(gca, 'TickLength', [0, 0]);
xlabel('time');
ylabel('Q');

end   % if fit_Q

