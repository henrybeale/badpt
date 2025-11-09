% particle filter with resampling, for use where certain parameters vary
% by categorical condition

classdef adaptive_1dSIR < handle
    properties
        N = 5e2
        resample_threshold = .5

        X
        Y
        ncond = 1
        parts
        likfun

        load = ''

        w
        theta
        liks
        dims

        stats
        plots
    end

    methods
        function obj = adaptive_1dSIR(varargin)
            % match input arguments with properties
            for n = 1: 2: length(varargin)
                assert(isprop(obj, varargin{n}), [varargin{n}, ' not found (spelling?)'])
                obj.(varargin{n}) = varargin{n+1};
            end

            % load a precomputed set of theta samples
            if ~isempty(obj.load)
                obj = getfield(load(obj.load),'obj');
            else
                % handle a new set of samples
                obj.theta.names = fieldnames(obj.theta);  % get names from struct
                for k = 1:numel(obj.theta.names)  % compile into new mat
                    obj.parts(:,k) = obj.theta.(obj.theta.names{k});
                end

                % size of stimuli x responses x conditions x particles
                obj.dims = [numel(obj.X),numel(obj.Y),obj.ncond,obj.N];

                disp('Computing likelihoods. This may take time ...'); tic;
                precomputeLikelihood(obj);
                toc
            end

            % reset weights
            obj.w = 1./ones(obj.N,1);
        end

        function precomputeLikelihood(obj)
            % copy some variables for parfor
            X = obj.X; 
            Y = obj.Y; 
            parts = obj.parts;
            dims = obj.dims;
            likfun = obj.likfun;

            liks = zeros(prod(dims),1);  % flattened array

            parfor k = 1:numel(liks)
                % get indices from the linear index
                ix = cell(length(dims),1);
                [ix{:}] = ind2sub(dims,k);
                
                % pull out stimulus and response for this iter
                [x, y] = deal(X(ix{1}), Y(ix{2})); 
                cond = ix{3};
                param = parts(ix{4},:);
            
                % evalute likelihood
                if obj.ncond > 1
                    liks(k) = likfun(y,x,param,cond);
                else
                    liks(k) = likfun(y,x,param);
                end
            end
            
            obj.liks = reshape(liks, obj.dims);
        end
   
        function [x] = selectStim(obj,futurecond)
            % compatibility with no conditions
            if nargin == 1 
                futurecond = 1; 
            end

            t0 = tic;
            % average entropy over responses
            H = zeros(size(obj.liks,1),1);
            
            % loop over response possibilities
            for k = 1:numel(obj.Y)
                % estimate future posteriors
                l = squeeze(obj.liks(:,k,futurecond,:));  % [stim(:) by particles] for Yi
                P = l.*obj.w';
                P = P./sum(P,2);
            
                p_data = l*obj.w;
                
                logp = log(P);
                logp(logp==-inf) = 0;  % correct for zero prob weights
                H = H + sum(-P.*logp,2).*p_data;
            end
            
            % select joint stimuli from concat stim array
            [~,ind] = min(H);
            x = obj.X(ind);

            obj.stats.compute_time = toc(t0);
        end

        function update(obj,x,y,cond)
            % compatibility with no conditions
            if nargin < 4 || isempty(cond)
                cond = 1; 
            end

            % find the stim array corresponding to stimuli
            [~,stim_ind] = min(abs(obj.X-x));
            resp_ind = find(y == obj.Y);

            % update particle weights
            obj.w = squeeze(obj.liks(stim_ind,resp_ind,cond,:)).*obj.w;
            obj.w = 1/sum(obj.w)*obj.w;  % normalise
            
            calc_stats(obj)  % keep neff, entropy updated

            % resample if neff falls below threshold
            if obj.stats.neff < obj.resample_threshold*obj.N
                resample(obj)
            end
        end

        function calc_stats(obj)
            % some stats that can be helpful
            obj.stats.entropy = -sum(obj.w.*log(obj.w));
            obj.stats.neff = 1./sum(obj.w.^2);
        end

        function est = estimate(obj)
            % return parameter means using particle weights
            est = obj.w'*obj.parts;
        end

        function [samples, sampleinds] = sample(obj, N)
            % sample from particles according to weights
            sampleinds = randsample(1:length(obj.w), N, true, obj.w);
            samples = obj.parts(sampleinds,:);
        end
        
        function resample(obj, N)
            % resample particles
            if nargin == 1
                N = obj.N;  % default to same N particles
            end

            [obj.parts, inds] = sample(obj, N);
            obj.liks = obj.liks(:,:,:,inds);
            obj.w = 1./ones(length(inds),1); % reset weights to uniform
            obj.N = length(obj.w);
        end        

        function save(obj, filename)
            save(filename, 'obj', '-v7.3','-nocompression')
        end

        function fig = plot_start_particles(obj, fig)
            f = figure; 
            for k = 1:numel(obj.theta.names)
                nexttile
                histogram(obj.theta.(obj.theta.names{k}), 90, 'edgecolor', 'none')
                title(obj.theta.names{k})
            end
            sgtitle('Prior samples')
            f.Children.TileSpacing = 'tight';
        end

        function create_tracking_figure(obj, true_param)
            % create handle to a figure that will be updated trialwise
            obj.plots.trackfig = figure;
            tiledlayout(2,max(4,numel(obj.theta.names)),'tilespacing','tight','tileindexing','rowmajor')

            names = {'entropy','effsamp','compute time','error'};
            if ~exist('true_param','var')
                names = names(1:end-1);
            else
                obj.plots.true_param = true_param;
            end

            clear obj.plots.ln
            obj.stats.trackdata = nan(1e3, numel(obj.theta.names)+numel(names));  % limit 1000 trials tracked then it fails
            obj.plots.trackind = 0;

            for k = 1:size(obj.stats.trackdata,2)
                nexttile
                obj.plots.ln(k) = plot(obj.stats.trackdata(:,k),'linewidth',1);
                if k <= numel(obj.theta.names)  % plot parameters
                    title(obj.theta.names{k})
                    if ismember('error', names)
                        xline(obj.plots.true_param(k),'linewidth',2);
                    end
                    obj.plots.xl(k) = xline(0, 'linewidth',1, 'color', 'r');
                else  % plot other stats
                    title(names(k-numel(obj.theta.names)))
                end
            end
        end

        function update_tracking_figure(obj)
            % increment index
            obj.plots.trackind = obj.plots.trackind + 1; 

            % estimate parameters
            ntheta = numel(obj.theta.names);
            obj.stats.trackdata(obj.plots.trackind,1:ntheta) = estimate(obj);
            samples = sample(obj,5e2);

            % track stats
            obj.stats.trackdata(obj.plots.trackind,ntheta+1) = obj.stats.entropy;
            obj.stats.trackdata(obj.plots.trackind,ntheta+2) = obj.stats.neff;
            try
                obj.stats.trackdata(obj.plots.trackind,ntheta+3) = obj.stats.compute_time;
            end
            
            % optional compute distance from known params
            if isfield(obj.plots, 'true_param')
                obj.stats.trackdata(obj.plots.trackind,end) = norm(obj.stats.trackdata(obj.plots.trackind,1:ntheta)-obj.plots.true_param);
            end
            
            figure(obj.plots.trackfig)
            for k = 1:size(obj.stats.trackdata,2)
                if k <= ntheta
                    [f,xi] = ksdensity(samples(:,k));
                    set(obj.plots.ln(k), 'XData', xi)
                    set(obj.plots.ln(k), 'YData', f) 
                    set(obj.plots.xl(k), 'Value', obj.stats.trackdata(obj.plots.trackind,k))
                else
                    if ~isprop(obj.plots.ln(k),'CData')
                        set(obj.plots.ln(k), 'YData', obj.stats.trackdata(1:obj.plots.trackind,k))
                    else
                        try
                            set(obj.plots.ln(k), 'CData', -obj.stats.H')
                        end
                    end
                end
            end
            drawnow
        end
    end

    methods(Static)
        function obj = loadobj(s)
            % allows loading
            obj = s;
        end
    end
end
    