% adaptive method for estimating psychometric function

classdef Badpt < handle

    properties
        prob
        stim
        parms
        parms_ID
        parm_to_marg
        fun
        resp_opts = 0:1 
        p  % prior
        P  % posterior
        pick_ix
        dims
        parm_ix
        prior = 'uniform'
        
        thresh = .5  % default values for starting
        slope = .3  
        guess_rate = .5
        lapse_rate = .01
        theta
    end

    methods
        function obj = Badpt(varargin)
            for n = 1: 2: length(varargin)
                assert(isprop(obj, varargin{n}), [varargin{n}, ' not found (spelling?)'])
                obj.(varargin{n}) = varargin{n+1};
            end

            % use a predefined probability function
            if isempty(obj.fun)  || strcmpi(obj.fun, 'gaussian')
                PF = @(th, x) th(4) + (1 - abs(th(3)) - th(4)) * normcdf(log10(x), log10(th(1)), th(2));  % pdf
                obj.fun = @(r, th, x) binopdf(r, 1, PF(th, x));
            end

            % any parameters not provided will be fixed at the default
            obj.theta = [obj.thresh, obj.slope, obj.lapse_rate, obj.guess_rate];
            
            % look for precomputed likelihoods
            if ischar(obj.parms)
                % load from file
                obj.prob = getfield(load(obj.parms), 'prob');
                obj.stim = getfield(load(obj.parms), 'stim');
                obj.parms = getfield(load(obj.parms), 'parms');  % load in the struct of data
                obj.dims = size(obj.prob, 1:length(fieldnames(obj.parms)));  % size of the parameter space
                obj.parms_ID = fieldnames(obj.parms);
            else
                % handle given parameters
                assert(mod(length(obj.parms),2) == 0, 'Key-value pairs in "parms" unmatched')
                obj.parms = struct(obj.parms{:}); 
                obj.parms_ID = fieldnames(obj.parms);
                
                % get the dimensions of the parameter space
                obj.dims = []; 
                for i = 1:length(obj.parms_ID)
                    f = obj.parms_ID{i};
                    obj.dims = [obj.dims, length(obj.parms.(f))];
                end
                
                disp('Computing likelihoods. This may take time ...'); tic;
                precomputeLikelihood(obj);
                toc
            end
            
            % create prior
            if strcmpi(obj.prior, 'uniform')
                obj.p = ones(obj.dims) / prod(obj.dims);
                obj.prior = obj.p; 
            end

            % create a variable to ease indexing later
            obj.parm_ix = repmat({':'}, 1, numel(obj.dims));
            
            % checks
            assert(~isempty(obj.stim), 'no stimulus array provided')
        end

        function precomputeLikelihood(obj)
            % precompute the likelihood of the parameters 
            
            % create a flattened parameter array
            xdim = [obj.dims, length(obj.stim), length(obj.resp_opts)];
            dims = obj.dims;  % just the parameters
            prob = zeros(xdim);           
            prob = reshape(prob, 1, []);
            
            % pull out some vars from the object (to please parfor)
            th = obj.theta;
            parms = obj.parms; 
            stim = obj.stim; 
            parms_ID = obj.parms_ID;
            fun = obj.fun; 

            parfor i = 1: length(prob)
                % get the indices of the original parameter arrays
                ix = cell(length(xdim),1);
                [ix{:}] = ind2sub(xdim, i);
                
                x = stim(ix{end-1});  
                r = obj.resp_opts(ix{end});  % resp 
                
                loop_th = th;
                
                for j = 1:length(parms_ID)
                    f = parms_ID{j}; 
                    loop_th(j) = parms.(f)(ix{j});
                end
                
                prob(i) = fun(r, loop_th, x);
            end
            
            obj.prob = reshape(prob, xdim);
            delete(gcp); 
        end

        function save(obj, f_name)
            prob = obj.prob;
            stim = obj.stim;
            parms = obj.parms;
            save(f_name, 'prob', 'stim', 'parms')
        end

        function lik = update(obj, resp, contr)
            % Compute the posterior parameter distribution given the
            % observed response. This posterior becomes the prior that is
            % when searching through the posteriors to get the next trial
            % stimulus. 

            % get the contrast if available from precomputed data
            if nargin < 3
                contr_ix = obj.pick_ix;  % stored from last selection
            else
                contr_ix = find(obj.stim == contr);  % check in stim array
            end

            % likelihood of observing resp given theta (parameters) and stim x
            if ~isempty(contr_ix)
                % use the precomputed likelihood
                resp_ix = find(obj.resp_opts == resp);
                lik = obj.prob(obj.parm_ix{:}, contr_ix, resp_ix);
            else
                % compute the likelihood of the parameters given a novel
                % contrast value. (VERY SLOW)
                fprintf('Computing likelihood, lookup failed\n')
                lik = reshape(zeros(obj.dims), [], 1); 
                ix = cell(length(obj.dims));  % indices for the parameters
                
                for i = 1:length(lik)
                    [ix{:}] = ind2sub(obj.dims, i); 
                    th = obj.theta; 
    
                    % set the parameter
                    for j = 1: length(obj.parms_ID)
                        f = obj.parms_ID{j}; 
                        th(j) = obj.parms.(f)(ix{j});
                    end
    
                    lik(i) = obj.fun(resp, th, contr);
                end

                lik = reshape(lik, obj.dims);
            end
            
            % overall probability of resp stim condition x 
            p_resp = sum(lik(:) .* obj.p(:));  % sum over theta
            
            % Posterior (Bayes rule) 
            obj.P = (lik / p_resp) .* obj.p; 
            
            % (P)osterior becomes next (p)rior
            obj.p = obj.P;
        end

        function stim_value = selectStim(obj, parm_to_marg)
            % Compute the expected entropy for each stimulus option and
            % return the value from the stimulus array with the minimum
            % entropy. Optionally marginalise any parameter from the
            % posterior distribution. 
            
            if nargin < 2
                parm_to_marg = obj.parm_to_marg;
            end
            
            % handle named parameters to marginalise
            if ischar(parm_to_marg) || iscell(parm_to_marg)
                parm_to_marg = find(contains(obj.parms_ID, parm_to_marg));
            end

            % compute the expected posterior for all stim values
            EH = zeros(1, length(obj.stim));  % to store expected entropy (H)
            
            for i = 1:length(obj.stim)
                for r = 1:2
                    % do as above (as if we are a trial ahead)
                    lik = obj.prob(obj.parm_ix{:},i,r);
                    p_resp = sum(lik(:) .* obj.p(:));  % use the updated prior
                    P_plus1 = lik / p_resp .* obj.p;

                    % marginalise nuisance parameters
                    if ~isempty(parm_to_marg)
                        P_plus1 = squeeze(sum(P_plus1, parm_to_marg));
                    end
                    
                    % calculate the entropy of this posterior
                    H = -sum(P_plus1(:) .* log(P_plus1(:))); 
                    EH(i) = EH(i) + H * p_resp;  % weight by the prob of resp.
                end
            end

            % select the stimulus value that with minimum expected entropy
            [~,obj.pick_ix] = min(EH); 
            stim_value = obj.stim(obj.pick_ix);
        end

        function resetPrior(obj)
            obj.p = obj.prior; 
        end

        function [est, arr] = estimateParms(obj)
            % Return mean of the parameter distributions. 
            est = struct; 
            arr = nan(1, length(obj.dims)); 
            for i = 1: length(fieldnames(obj.parms))
                f = obj.parms_ID{i};
                
                % marginal distribution of this parameter
                marg_ix = find(~contains(obj.parms_ID, f));
                marg = squeeze(sum(obj.P, marg_ix));
                
                est.(f) = sum(obj.parms.(f)' .* marg(:));
                arr(i) = est.(f);
            end
        end
    end
end




