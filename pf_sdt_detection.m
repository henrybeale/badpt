function [pdf, prob, dprime, target_x] = pf_sdt_detection(y, x, params, target_d)
% A signal detection model of contrast detection with parameters for
% 'threshold', slope, criterion, and lapse rate. 
% 
% d' is modelled as a Naka-Rushton function of x (e.g. contrast). 

% parameters 
Rmax = 5; 
c50 = params(1); 
n = exp(params(2)); 
crit = params(3); 
eps = params(4); 

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

if nargin >= 4 && ~isempty(target_d)
    assert(target_d < Rmax, "Targeted d' threshold must be less than Rmax=5")
    target_x = c50*(target_d ./ (Rmax - target_d)).^(1./n);
else
    target_x = []; 
end
