% this script will show you how to use an adaptive psychophysical procedure
% in a detection task. the relevant code should be entered in your
% experiment code. 

%% let's inspect the psychometric function used in detection
% ground truth parameters for simulation
true_param = [0.05, .1, log(3), 3, 0.05]; % [threshold intercept, threshold conditional, log(slope), criterion, lapse rate]

% function
targD = 1.5;  % optional; get contrast for a specific d' value
x = linspace(0,1); 

cond_vals = [0,1];

clf
nexttile
hold on 

for n = 1:numel(cond_vals)
    [~,y,d,targX] = pf_sdt_detection_2c([], x, true_param, cond_vals(n), targD);
    plot(x,y)
    title('Psychometric function'); ylabel('Prop. correct')
    ylim([0,1])
end
legend({'c=0','c=1'})

nexttile
hold on 
for n = 1:numel(cond_vals)
    [~,y,d,targX] = pf_sdt_detection_2c([], x, true_param, cond_vals(n), targD);
    plot(x,d)
    title("d'"); ylabel("d'"); xlabel('Contrast')
end

%% (1) SETUP the adaptive particle filter 
% we first need to setup the stimulus and response values 
X = linspace(0, 1, 100);   % stimuli (discrete grid)
Y = [0, 1];  % response outcomes 
cond_vals = [0,1];

% then draw some starting particles for the algorithm
N = 5e4; 

theta = struct; 
theta.base_threshold = unifrnd(0,.5,[N,1]); % threshold, c50, for c=0
theta.change_in_threshold = unifrnd(-.3,.3, [N,1]); % threshold change (in probit(c50)) when c=1
theta.logslope = unifrnd(log(.1), log(10), [N,1]);
theta.crit = unifrnd(-3,3,[N,1]);
theta.lapse = unifrnd(0,.2,[N,1]);

% enter these into the MATLAB class and it will precompute some parameter
% likelihoods that are needed during usage. it will use parallel computing
% for speed.
S = adaptive_1dSIR('conds', cond_vals, 'theta', theta, 'X', X, 'Y', Y, 'likfun', @pf_sdt_detection_2c)
save(S, './detection_particles_2cond')

% or load previously computed
S = adaptive_1dSIR('load', './detection_particles_2cond')

% plot the starting particles
plot_start_particles(S)

%% (2) RUN on simulated responses
% setup:
n_trials = 600; 
n_adaptive_trials = 160;  % how many 'staircase' trials to use at the start
target_d = 1.5;
stim = nan(1,n_trials);
resp = nan(1,n_trials);
cond = randsample(repelem([0,1], 1, n_trials/2), n_trials, false);  % create randomised conditions

% ---- DO THIS BEFORE EXPERIMENT----
% reload particles
S = adaptive_1dSIR('load', './detection_particles');
param_history = nan(n_trials,S.num_params);
% create_tracking_figure(S)  % experimental

for n = 1:n_trials
    % ---- DO THIS BEFORE TRIAL PRESENTATION/DRAWING BEGINS ----
    if n <= n_adaptive_trials
        stim(n) = selectStim(S, cond(n));  % adaptive stimulus selection
    else
        % use another procedure for stimulus 
        if mod(n,2)  % every second trial present a stimulus
            % compute the stimulus for a target d' value
            params = estimate(S); 
            [~,~,~,stim(n)] = pf_sdt_detection_2c([],[],params,cond(n),target_d);
        else
            stim(n) = 0;  % stimulus absent
        end
    end

    % trial presentation and response collection here ... 
    [~,p] = pf_sdt_detection_2c([], stim(n), true_param, cond(n));  % use ground-truth to generate response
    resp(n) = binornd(1, p);

    % ---- DO THIS BEFORE NEXT TRIAL ----
    update(S, stim(n), resp(n), cond(n))
    param_history(n,:) = estimate(S);  % get the latest parameter estimates
    % update_tracking_figure(S) % experimental
end

% note that you can always update function tracker on any trial, even if
% you decide to stop the adaptive 'staircase' phase of the experiment. this
% will keep the posterior estimate up to date and can improve your
% thresholds. 
% 
% you can also always use the latest threshold estimate to base your
% stimulus on.

%% plot the simulation results
clf 
nexttile
hold on 
trl = 1:n_trials;
for k = 1:2
    scatter(trl(resp==(k-1)), stim(resp==(k-1)))
end
xline(n_adaptive_trials)
legend({'no','yes'})
xlabel('Trial')
ylabel('Stimulus')
title('Stimulus and response history')

for n = 1:numel(cond_vals)
    cval = cond_vals(n);

    nexttile
    hold on 
    xx = linspace(0,1);  % contrast 
    
    % true function
    [~,y_true,d_true] = pf_sdt_detection_2c([],xx, true_param, cval);
    plot(xx,y_true,'linewidth',2)
    
    % posterior estimate 
    [~,y_post,d_post] = pf_sdt_detection_2c([],xx,estimate(S),cval);
    plot(xx,y_post,':','linewidth',2)
    
    % perform a quick maximum likelihood fit
    p0 = true_param+rand(1,S.num_params)*.1;  % start search around true parameters
    fn = @(P) -sum(log(pf_sdt_detection_2c(resp, stim, P, cval)),'all');  % objective function
    [P, neg_log_lik] = fminsearch(fn, p0);
    
    [~,y_ml,d_ml] = pf_sdt_detection_2c([], xx, P, cval); 
    plot(xx,y_ml,'--','linewidth',1)
    
    % binned data with Gaussian window 
    xbins = linspace(0,1,30);
    c_ix = cond==cval;
    sigma = .01;  % gaussian width
    W = normpdf(stim(c_ix)-xbins',0, sigma);
    W = W ./ sum(W,2); 
    y_est = W*resp(c_ix)';
    ess =  1./ sum(W.^2,2);  % effective sample size in bin
    ess = ess./max(ess)*100;  % normalise for plot
    
    scatter(xbins, y_est, ess, 'filled','markerfacealpha',.7)
    
    legend({'true','final posterior','maximum likelihood','binned data'},'location','best')
    title(sprintf('Psychometric function, cond = %d', cval))
    ylabel('Probability correct')
    xlabel("Contrast")
end

% nexttile
% hold on 
% plot(xx,d_true,'linewidth',2)
% plot(xx,d_post,':','linewidth',2)
% plot(xx,d_ml,'--','linewidth',1)
% 
% % yl = (y_est - P(4)/2) ./ (1 - P(4));
% % yl = min(max(yl, eps), 1 - eps);   % clamp to avoid Inf from norminv
% % d_est = norminv(yl) - norminv(yl(1));
% % scatter(xbins, d_est, ess, 'filled')
% % xline(true_param(1));
% % xline(P(1),'--')
% 
% title("d'")
% ylabel("d'")
% xlabel('Contrast')

%% plot history of parameters over trials
clf 
for n = 1:S.num_params
    nexttile
    plot(param_history(:,n));
    yline(true_param(n),'r')
    title(S.theta.names{n})
end
sgtitle('Parameter estimate history')

%% draw samples from the importance weights
% this shows the posterior distribution at the most recent update(S)
samples = sample(S,1000);

clf
for n = 1:4
    [pdf,xx] = ksdensity(samples(:,n));
    nexttile
    plot(xx,pdf)
    xline(true_param(n),'r')
    xline(mean(samples(:,n)),'--')
    title(S.theta.names{n})
end
legend({'density','true','mean'},'location','best')
sgtitle('Sampled particles')
