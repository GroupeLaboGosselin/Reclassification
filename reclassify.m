function [accuracy_reclass, stats] = reclassify(accuracy, reclass_evidence)

% reclassify. This function reclassifies correct responses that are likely to be correct by chance as incorrect responses in 
% discrimination experiments using some evidence. We've recomputed 140 classification images with accuracies reclassified by 
% response time, for example, and increased SNR by more than 10% on average with little bias (Gosselin et al., submitted).
%
% Usage: 
%
%   [accuracy_reclass, stats] = reclassify(accuracy, reclass_evidence);
%
% The function takes 2 inputs: _accuracy_, a vector equal to 0 when the response to a trial was incorrect and to 1 when it 
% was correct; and _reclass_evidence_, a vector of the same length as _accuracy_ equal to a reclassification evidence for each 
% corresponding trial such as the response times. Note that the average of the evidence for incorrect responses ca be either 
% greater or smaller than the average of the evidence for correct responses but it must be different. The The function has 2 
% outputs: _accuracy_reclass_, a vector of the same length as _accuracy_ equal to 0 when the response to a trial was incorrect 
% or when a correct response to a trial was reclassified as incorrect, and to 1 when the response to trial was correct and 
% wasn't reclassified as incorrect; and _stats_ a structure with 4 fields: _reclass_evidence_criterion_, a reclassification 
% evidence such that when _reclass_evidence_polarity_ * _reclass_evidence_ > _reclass_evidence_polarity_ * _reclass_evidence_criterion_ 
% a correct response was reclassified as incorrect; _reclass_evidence_polarity_, either 1 or -1 and indicating how to interpret 
% the criterion; _reclass_index_, an index of the correct responses reclassified as incorrect; _reclass_efficiency_, the estimated 
% proportion of true correct and incorrect responses minus false correct and incorrect responses following reclassification; and _reclass_gain_, the ratio between _reclass_efficiency_ 
% and the efficiency prior to reclassification. Note that sqrt(_reclass_gain_) provides an approximation of the expected SNR gain.
%
% Gosselin, F., Daignault, V., Larouche, J.-M. & Caplette, L. (submitted). Reclassifying guesses to increase signal-to-noise ratio 
% in psychophysical experiments.
%
% Frederic Gosselin, 01/06/2020
% frederic.gosselin@umontreal.ca
%
% Minor modifications by Laurent Caplette, 17/08/2020
% laurent.caplette@yale.edu

    
% check that accuracy is composed of only zeros and ones
if length(unique(accuracy))~=2 || ~all(unique(accuracy)==[0 1])
    error('''accuracy'' variable must be composed of zeros and ones')
end

% transposes 1 variable if both are not in the same orientation
if all(size(accuracy')==size(reclass_evidence))
    accuracy = accuracy';
end

% checks that variables are of the same size
if ~all(size(accuracy)==size(reclass_evidence))
    error('''accuracy'' and ''reclass_evidence'' must have the same size')
end

% reverses the reclass_evidence if necessary
polarity = 1;                                                                                   % default reclass_evidence polarity
if (mean(reclass_evidence(find(accuracy==0)))-mean(reclass_evidence(find(accuracy==1))))<0,     % the reclass_evidence is greater for incorrect than correct trials
    polarity = -1;                                                                              % change reclass_evidence polarity
end
reclass_evidence = polarity * reclass_evidence;                                                 % the reclass_evidence multiplied by its polarity

% calculates histograms
nb_std = 2;
outliers = reclass_evidence > mean(reclass_evidence)+nb_std*std(reclass_evidence);              % temporary outliers to help frame the histogram
[n_reclass_evidence, bins] = histcounts(reclass_evidence(~outliers), 'BinMethod', 'fd');        % uses the Freedman-Diaconis rule for bin width
bin_width = bins(2)-bins(1);                                                                    % bin_width
bins = bins(1):bin_width:ceil(max(reclass_evidence)/bin_width)*bin_width;                       % complete reclass_evidence range, including outliers
index_correct = find(accuracy == 1);                                                            % correct response index
correct_reclass_evidence = reclass_evidence(index_correct);                                     % correct response reclass_evidences
n_correct = histcounts(correct_reclass_evidence, bins);                                         % correct reclass_evidences histograms
incorrect_reclass_evidence = reclass_evidence(find(accuracy==0));                               % incorrect response reclass_evidences
n_incorrect = histcounts(incorrect_reclass_evidence, bins);                                     % incorrect reclass_evidences histogram; this is also the false correct reclass_evidences histogram

% calculates frequency distributions
x = (bins(1:end-1)+bins(2:end))/2;                                                              % centers of the histogram bins
s_x = min(x):.01:max(x);                                                                        % fine histogram bins for interpolation
for ii = 1:numel(bins)-1,                                                                       % replaces histogram bin centers by histogram bin averages whenever possible
    ind = find(reclass_evidence>=bins(ii)&reclass_evidence<bins(ii+1));
    if not(isempty(ind)),
        x(ii) = mean(reclass_evidence(ind));
    end
end
s_n_correct = interp1(x, n_correct, s_x, 'spline');                                             % spline interpolation of n_true_correct reclass_evidence frequency distribution
s_n_incorrect = interp1(x, n_incorrect, s_x, 'spline');                                         % spline interpolation of n_incorrect reclass_evidence frequency distribution

s_n_true_correct = s_n_correct-s_n_incorrect;                                                   % interpolated true correct reclass_evidence frequency distribution
%s_n_true_correct = max(s_n_true_correct, 0);                                                    % set minimum to 0

% finds the best reclass_evidence criterion
N = sum(s_n_correct)+sum(s_n_incorrect);                                                        % number of points in all interpolated frequency distributions; general case
I_o = sum(s_n_incorrect);                                                                       % number of points in interpolated n_incorrect frequency distribution
cCR = cumsum(s_n_true_correct);                                                                 % cumulative interpolated true correct reclass_evidence frequency distribution 
cM = cumsum(s_n_incorrect);                                                                     % cumulative interpolated false correct reclass_evidence frequency distribution 
s_efficiency = (4*I_o - N + 2*(cCR - cM)) / N;                                                  % interpolated efficiency as a function of reclass_evidence reclassificatyion criterion; general case
s_ind = find(s_efficiency==max(s_efficiency), 1, 'first');                                      % finds maximum reclassification efficiency
reclass_evidence_criterion = s_x(s_ind);                                                        % chosen reclass_evidence criterion

% reclassifies correct responses as incorrect responses
accuracy_reclass = double(accuracy);                                                            % initialize with old accuracy
reclass_index = find((accuracy == 1) & (reclass_evidence > reclass_evidence_criterion));        % which correct response should be reclassified as an incorrect
accuracy_reclass(reclass_index) = 0;                                                            % reclassification per se

% some statistics
stats.reclass_evidence_polarity = polarity;                                                     % the reclassified correct responses are smaller than the criterion
stats.reclass_evidence_criterion = polarity * reclass_evidence_criterion;   
stats.reclass_index = reclass_index;
stats.reclass_efficiency = s_efficiency(s_ind);
stats.reclass_gain = s_efficiency(s_ind) / (1-2*(1-mean(accuracy))); 

%% for making figures only
% figure, plot(s_x, s_efficiency,'-', reclass_evidence_criterion, max(s_efficiency), '*')
% figure, plot(s_x, s_n_incorrect, 'r.', s_x, s_n_true_correct, 'g*', s_x, s_n_correct, 'k')

