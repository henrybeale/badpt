function [p,pr_corrected] = psycfun_pdf(y, c, params, cond)

% parameters
s = params(1); % contrast at d' = 1
n = params(2); 
L = params(3); 
eps = params(4);

% fix d' response maximum
R = 5; 

% psyc fun in d'
d = R*c.^n / (c.^n + R*s^n - s^n);  

% proportion correct
pr = 1-normcdf(L-d); 

% correction for lapse errors
pr_corrected = eps + (1-2*eps)*pr; 

% return probability of the response outcome 
p = binopdf(y, 1, pr_corrected);
return 


% function [p] = psycfun_pdf(y, c, R, s, n, L)
% 
% if nargin == 3  % redistribute params from vector 
%     s = R(2); 
%     n = R(3); 
%     L = R(4); 
%     R = R(1); 
% end
% 
% d = R*c.^n./(c.^n + s.^n);  % psyc fun in d'
% pr = 1-normcdf(L-d); % proportion correct
% 
% p = binopdf(y, 1, pr);  % return pdf of the response given the CRF 
% return 