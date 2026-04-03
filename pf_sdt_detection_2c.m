function [pdf, prob, dprime, target_x] = pf_sdt_detection_2c(y, x, params, cond, target_d)
% A signal detection model of contrast detection with parameters for
% 'threshold', slope, criterion, and lapse rate. Takes a condition variable
% for use in a linear model of parameters. Currently, the linear model is
% just for threshold, you would need to modify it below for any other
% combination of parameters (expanding the params vector).
% 
% d' is modelled as a Naka-Rushton function of x (e.g. contrast). 

% check values are set up correctly for linear model 
assert(any(ismember(cond, [0,1])), 'condition input must be in [0,1]')

% parameters 
Rmax = 5; % fixed
a0 = params(1); % threshold intercept
a = params(2);  % threshold adjustment, conditional on cond==1
n = exp(params(3)); % transform to positive (input is log slope)
crit = params(4); 
eps = params(5); 

% make a linear model of the threshold parameter, that depends on condition
c50 = normcdf(norminv(a0) + cond*a);  % linear model in unconstrained space

% naka-rushton d prime function
dprime = Rmax*x.^n ./ (x.^n + c50.^n); 

% SDT model for probability correct
prob = eps + (1-2*eps)*(1-normcdf(crit-dprime));

% compute the likelihood of response outcome
if ~isempty(y)
    pdf = binopdf(y, 1, prob);
else
    pdf = []; 
end

if nargin >= 5 && ~isempty(target_d)
    assert(target_d < Rmax, "Targeted d' threshold must be less than Rmax=5")
    target_x = c50*(target_d ./ (Rmax - target_d)).^(1./n);
else
    target_x = []; 
end
