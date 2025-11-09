
%% inspect the function 
true_param = [0, log(2), .1]; % [bias, logslope, lapse]

x = linspace(-3,3); 
[~,y] = psycfun_2afc([], x, true_param);

clf
plot(x,y)

%% setup adaptive sequential importance sampling 
X = linspace(-2,2,60);  % discrete stimuli chosen by the algorithm
Y = [0, 1];  % response options

% sample from prior
N = 5e4; 
slope_limits = [log(.1), log(20)];

th = struct; 
th.bias = randn(N,1)*2; 
th.logslope = randn(N,1)*2 + log(1.5);
th.lapse = rand(N,1)*.3;

S = adaptive_1dSIR('N',N,'theta',th,'X',X,'Y',Y,'likfun',@psycfun_2afc)
save(S, './tmp')

plot_start_particles(S)

%% simulate trials
niter = 300;

% precompute random stimuli 
stim = randn(1,niter);
y = nan*stim;

S = adaptive_1dSIR('load', './tmp');
stim_hist = nan(niter,3);

% close all
% create_tracking_figure(S, true_param)figure

for n = 1:niter
    % prior to trial: select adaptive stimulus
    [x] = selectStim(S);  % adaptive look-ahead finds most informative stimulus
    % stim(n) = x;

   
    % evalute true psyc func
    [~,p] = psycfun_2afc([], stim(n), true_param);
    y(n) = binornd(1, p);  % draw random value

    % update weights 
    update(S, stim(n), y(n))  % x values will be mapped to nearest point in X, which allows random stimuli to be used
    % update_tracking_figure(S)
    stim_hist(n,:) = estimate(S);
end

%% plot history
stim_hist = max(-5,min(5,stim_hist));
clf
for n = 1:3
    nexttile
    tmp = stim_hist(:,n);
    tmp(abs(zscore(tmp))>3) = nan;  % censor data at resampling steps
    plot(1:niter, tmp)
    yline(true_param(n),'color','r')
    title(S.theta.names{n})
    xlabel('Trial')
end
sgtitle('Parameter estimates')

%% smooth proportion correct using gaussian window bin
xbins = linspace(-3,3,20);
sigma = .1; 

% matrix of binned contrast weights
W = normpdf(stim-xbins',0,sigma);
W = W ./ sum(W,2);
y_smooth = W*y';
ess = 1./ sum(W.^2,2);  % effective sample size (e.g. bin count)

ess = ess./max(ess)*100;

% maximum likelihood fitting 
p0 = true_param+randn(1,3)*.1;
% p0(3) = exp(p0(3));
fn = @(P) -sum(log(psycfun_2afc(y,stim,P)));  % negatie log likelihood
[P, nll] = fminsearch(fn, p0);

clf
nexttile
scatter(xbins, y_smooth,ess)
hold on 
[~,pr] = psycfun_2afc([],xbins,true_param);
plot(xbins,pr)
[~,pr] = psycfun_2afc([],xbins,S.estimate);
plot(xbins,pr, 'k--')

[~,pr] = psycfun_2afc([],xbins,P); 
plot(xbins,pr,'g')
legend({'binned data','true model','final adaptive fit','max lik fit'},'location','best')
title('Psychometric function')

% stimulus history
nexttile
plot(stim,'.')
xlabel('Trial')
ylabel('stimulus')
title('stimulus history')

%% draw samples from the importance weights
% this shows the posterior distribution
samples = sample(S,500);

clf
for n = 1:3
    [pdf,xx] = ksdensity(samples(:,n));
    nexttile
    plot(xx,pdf)
    xline(true_param(n),'r')
    xline(mean(samples(:,n)),'--')
    title(S.theta.names{n})
end
legend({'density','true','mean'},'location','best')