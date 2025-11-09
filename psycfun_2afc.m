function [pdf, prob] = psycfun_2afc(y, x, params)
% use a cumulative gaussian psychometric function


% parameters 
mu = params(1);
slope = exp(params(2)); 
lapse = (params(3)); 

prob = lapse + (1-2*lapse).*normcdf(slope.*(x-mu));

% evalute pdf 
pdf = [];
if ~isempty(y)
    pdf = binopdf(y,1,prob);
end
