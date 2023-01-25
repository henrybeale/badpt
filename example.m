
%% this is how you initialise it for use
% create parameter space
thr = linspace(.1, .7, 40); 
slo = linspace(1/100, 1/2, 45);
lam = linspace(.01, .4, 20);  % lapse rate
gam = linspace(.3, .7, 15);  % guess rate

% create a discrete stimulus space
stim = linspace(.1, .8, 120);

% load the Badpt class and pass through the . it will take some time to
% compute the likelihood of each stimulus value under all combinations of
% the parameters
B = Badpt('parms', {'thr', thr, 'slo', slo, 'lam', lam, 'gam', gam}, ...
            'stim', stim);

% the class has it's own function that will save the precomputed lieklihood
% space, so that it can just be loaded on every new session. (note this is
% needed as a starting place for the algorithm and contains no updated info
% from each subject).
save(B, './data/precompBadpt1.mat')

% on every other session with the same stimulus and parameter space, you
% can just initialise the class and load the precomputed likelihood space: 

% B = Badpt('parms', './data/precompBadpt1.mat');  

%% usage on each trial
% before starting anything you can select a stimulus value that will be
% informative. it does this by looking ahead at each of the possible
% stimulus values and the one that results in the most informative change
% in posterior (averaging over either response outcome, hit/miss). for this
% reason, it can take time (~1s) depending on the size of the parameter and
% stimulus space.
stimulus_value = selectStim(B); 

% once this stimulus has been shown and a response is gained, you can
% update the algorithm after each trial. NOTE this is relatively quick and
% can be done on each trial even after the 'titration' period has ended.
update(B, response, stimulus_value);

% you can also get an online estimate of the most probable parameters (in
% either a described struct summary or in numeric form).
[struct_summary, parameter_array] = estimateParms(B);

%% marginalising parameters
% you can choose to marginalise certain parameters 
stimulus_value = selectStim(B, {'lam', 'gam'});

%% transferring the updated posterior between sessions
% at the end of a session we can simply save the updated posterior as a
% regular matlab file.
posterior = B.p;
save('./sub-01/posterior_ses-01.mat', 'posterior')

% and then load it to continue updating it in a new session 
load('./sub-01/posterior_ses-01.mat', 'posterior')
B.p = posterior;

% if you have loaded this but want to reset the prior to be uniform then
% run this: 

% resetPrior(B);
