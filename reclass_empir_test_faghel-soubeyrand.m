% Matlab script for testing of reclassification procedure introduced in 
% Gosselin, F, Daigneault, V., Larouche, J.-M. & Caplette, L. (submitted). Reclassifying guesses to increase signal-to-noise ratio in psychological experiments.
% with the database of
% Faghel-Soubeyrand, S., Dupuis-Roy, N. & Gosselin, F. (2019). Inducing the use of right eye enhances face-sex categorization performance. Journal of Experimental Psychology: General, 148, 1834-1841.
% (available at https://osf.io/jfz68).
% 
% The reclassification procedure that can increase signal-to-noise ratio in psychological experiments that use accuracy as a selection 
% variable for another dependent variable. It relies on the fact that some correct responses result from guesses and reclassifies them as 
% incorrect responses using a trial-by-trial reclassification evidence such as response time. It selects the optimal reclassification 
% evidence criterion beyond which correct responses should be reclassified as incorrect responses assuming that all incorrect responses 
% result from guesses.
%
% Here, we test the reclassification procedure with the Faghel-Soubeyrand et al. (2019) dataset using RT as reclassification evidence. 
% We compute the SNR gain of the responses reclassified following the application of the reclassification =procedure relative to the 
% original responses. We also compute the SNR gain associated with reclassifying the unique mean percentage of correct responses 
% reclassified as incorrect responses by the reclassification procedure to all participants. Finally, we compute the SNR gains for three 
% other procedures discussed in the "Comparison with other procedures" section of the paper: Tukey?s, Wise and Ma?s and the mean RT + 2 RT 
% standard deviation. We use only ?greater than? criteria, just like in the reclassification procedure. Furthermore, we tried both 
% rejecting and reclassifying the responses above the criteria. In all cases, reclassification outperforms rejection. Therefore, we only 
% report the reclassification SNR gains below.
%
% Reclassified accuracy leads to an actual SNR gain of 113.46%. Accuracies resulting from reclassifying the average proportion of individual 
% responses reclassified by our reclassification procedure ? the slowest 9.97% correct responses ? leads to a smaller but nonetheless 
% important SNR gain of 110.22%. Both these SNR gains are greater than the SNR gains associated with Tukey?s fences (107.55%), Wise and 
% Ma?s criterion (109.20%) and the mean + 2 st. dev. criterion (106.04%)).
%
% High z-score values in the group classification images (CI) computed from different weights for the Faghel-Soubeyrand et al. (2019) 
% dataset are also shown. The inferno color map that we use is perceptually uniform. The face outline is shown to help with interpretation.


%% loads data
load('faghel_soubeyrand2019.mat');  % load data from Faghel-Soubeyrand et al. (2019)
nb_blocks = 3;                      % number of blocks of 100 trials completed by subjects pre-training, i.e. the first 3 blocks
nb_subjects = 140;                  % number of subjects that did these pre-training blocks
nb_trials_per_block =100;           % number of trials per block

%% initialises the differenr classification images (Ci)
xySize = 128;                                       % Ci width and height in pixels
Ci_vect_standard = zeros(nb_subjects, xySize^2);    % based on the original accuracy
Ci_vect_reclass = zeros(nb_subjects, xySize^2);     % based on the reclassification procedure criterion
Ci_vect_prop = zeros(nb_subjects, xySize^2);        % based on the unique mean percentage of correct responses reclassified
Ci_vect_tukey = zeros(nb_subjects, xySize^2);       % based on Tukey's fence (Tukey, 1977)
Ci_vect_wise_and_ma = zeros(nb_subjects, xySize^2); % based on Wise and Ma's criterion (Wise & Ma, 2012)
Ci_vect_reject = zeros(nb_subjects, xySize^2);      % based on the the mean RT + 2 RT standard deviation criterion


%% computes classsification images participant per participant
for subject = 1:nb_subjects,
    
    subject

    %% prepares the data
    tmp = squeeze(double(data.acc(subject,:,:))); % all accuracy data
    tmprt = squeeze(double(data.rts(subject,1:nb_blocks,:))); % all RT data
    RT = zeros(1,nb_blocks*nb_trials_per_block); % initializes RT vector for first 3 blocks
    accuracy = zeros(1,nb_blocks*nb_trials_per_block); % initiliazes accuracy vector for first 3 blocks
    
    % constructs RT and accuracy vectors for first 3 blocks
    for ii = 1:nb_blocks,
        accuracy(1,(ii-1)*nb_trials_per_block+1:ii*nb_trials_per_block) = tmp(ii,:);
        RT(1,(ii-1)*nb_trials_per_block+1:ii*nb_trials_per_block) = tmprt(ii,:);
    end
    
    % constructs bubbles masks matrix for first 3 blocks
    tmp = squeeze(double(data.bub_mask(subject,:,:,:))); % all bubbles masks data
    X = zeros(nb_blocks*nb_trials_per_block, xySize^2); % initializes bubbles masks matrix for first 3 blocks
    for ii = 1:nb_blocks,
        X((ii-1)*nb_trials_per_block+1:ii*nb_trials_per_block,:) = tmp(ii,:,:);
    end
    std_tmp = std(X,0,2);   
    X = (X - repmat(mean(X,2), 1, xySize^2)) ./ repmat(std_tmp, 1, xySize^2); % tranforms X in z-scores
    X(isnan(X)) = 0; % removes Nan's due to 0 bubbles trials


    %% computes vectorised classification images (Ci)    
    % Ci based on original accuracy
    y_standard = (accuracy-mean(accuracy))/std(accuracy); % transforms accuracy in z-scores
    Ci_vect_standard(subject,:) = y_standard*X/sqrt(length(accuracy)); % constructs vectorized classification image and completes its transformation in z-scores
    
    % Ci based on the optimal reclassification procedure criterion
    [accuracy_reclass stats] = reclassify(accuracy, RT); % runs the reclassification procedure
    y_reclass = (accuracy_reclass - mean(accuracy_reclass)) / std(accuracy_reclass);
    Ci_vect_reclass(subject,:) = y_reclass*X/sqrt(length(accuracy));
%    prop_reclass(subject) = numel(stats.reclass_index)/(nb_trials_per_block*nb_blocks); % used to calculate the unique mean percentage of correct responses reclassified mean percent of reclassification

    % Ci based on the unique mean percentage of correct responses reclassified mean percent of reclassification
    accuracy_prop = accuracy;
    mean_percentage = 0.0997; % mean percentage of correct responses reclassified mean percent of reclassification; equal to mean(prop_reclass)
    rt_criterion = quantile(RT, 1-mean_percentage); % reclassification criterion
    index = find(RT>=rt_criterion);
    accuracy_prop(index) = 0;
    y_prop = (accuracy_prop - mean(accuracy_prop)) / std(accuracy_prop);
    Ci_vect_prop(subject,:) = y_prop*X/sqrt(length(accuracy));
    
    % Ci based on the Wise & Ma (2012) criterion that is, > 90% of mean
    accuracy_wise_and_ma = accuracy;
    rt_criterion = 1.9*mean(RT); % Wise & Ma's criterion
    index = find(RT>=rt_criterion);
    accuracy_wise_and_ma(index) = 0;
    y_wise_and_ma = (accuracy_wise_and_ma - mean(accuracy_wise_and_ma)) / std(accuracy_wise_and_ma);
    Ci_vect_wise_and_ma(subject,:) = y_wise_and_ma*X/sqrt(length(accuracy));
    
    % Ci based on on the the mean RT + 2 RT standard deviation criterion
    accuracy_reject = accuracy;
    rt_criterion = mean(RT) + 2 * std(RT);
    index = find(RT>=rt_criterion);
    accuracy_reject(index) = 0;
    y_reject = (accuracy_reject - mean(accuracy_reject)) / std(accuracy_reject);
    Ci_vect_reject(subject,:) = y_reject*X/sqrt(length(accuracy));
    
    % Ci based on Tukey's fence (Tukey, 1977)
    accuracy_tukey = accuracy;
    rt_criterion = quantile(RT, .75) + 1.5 * iqr(RT);
    index = find(RT>=rt_criterion);
    accuracy_tukey(index) = 0;
    y_tukey = (accuracy_tukey - mean(accuracy_tukey)) / std(accuracy_tukey);
    Ci_vect_tukey(subject,:) = y_tukey*X/sqrt(length(accuracy));

end


%% computes smooth group classification images and their signal-to-noise ratio (SNR) gains

% defines the smoothing kernel
sigma = 5; % standard deviation of Gaussian smoothing kernel
TNoyau = 6*sigma;
gauss = fspecial('gaussian',ceil(TNoyau),sigma); % computes Gaussian smoothing kernel

% computes smooth 2D group classification image based on original accuracy
Ci = reshape(squeeze(sum(Ci_vect_standard, 1)), xySize, xySize)/sqrt(nb_subjects); % computes a 1D group Ci and reshapes it into a 2D group Ci
sCi = filter2(gauss, Ci); % smooth the 2D group Ci
sCi_standard = sCi/sqrt(sum(gauss(:).^2)); % completes the transformation of the smooth group Ci in z-scores

% computes smooth 2D group classification image based on the optimal reclassification procedure criterion
Ci = reshape(squeeze(sum(Ci_vect_reclass, 1)), xySize, xySize)/sqrt(nb_subjects);
sCi = filter2(gauss, Ci);
sCi_reclass = sCi/sqrt(sum(gauss(:).^2)); % transforms the group Ci in z-scores
snr_gain_reclass = std(sCi_reclass(:))/std(sCi_standard(:)) % gain observed in the Ci based on the optimal reclassification procedure relative to the Ci based on original accuracy

% computes smooth 2D group classification image based on the unique mean percentage of correct responses reclassified
Ci = reshape(squeeze(sum(Ci_vect_prop, 1)), xySize, xySize)/sqrt(nb_subjects);
sCi = filter2(gauss, Ci);
sCi_prop = sCi/sqrt(sum(gauss(:).^2));
snr_gain_prop = std(sCi_prop(:))/std(sCi_standard(:)) % gain observed in the Ci based on the mean percentage of correct responses reclassified relative to the Ci based on original accuracy

% computes smooth 2D group classification image based on the Wise & Ma (2012) criterion that is, > 90% of mean
Ci = reshape(squeeze(sum(Ci_vect_wise_and_ma, 1)), xySize, xySize)/sqrt(nb_subjects);
sCi = filter2(gauss, Ci);
sCi_wise_and_ma = sCi/sqrt(sum(gauss(:).^2));
snr_gain_wise_and_ma = std(sCi_wise_and_ma(:))/std(sCi_standard(:)) % gain observed in the Ci based on the Wise & Ma (2012) relative to the Ci based on original accuracy

% computes smooth 2D group classification image based on on the the mean RT + 2 RT standard deviation criterion
Ci = reshape(squeeze(sum(Ci_vect_reject, 1)), xySize, xySize)/sqrt(nb_subjects);
sCi = filter2(gauss, Ci);
sCi_reject = sCi/sqrt(sum(gauss(:).^2));
snr_gain_reject = std(sCi_reject(:))/std(sCi_standard(:)) % gain observed in the Ci based on on the the mean RT + 2 RT standard deviation criterion relative to the Ci based on original accuracy

% computes smooth 2D group classification image based on Tukey's fence (Tukey, 1977)
Ci = reshape(squeeze(sum(Ci_vect_tukey, 1)), xySize, xySize)/sqrt(nb_subjects);
sCi = filter2(gauss, Ci);
sCi_tukey = sCi/sqrt(sum(gauss(:).^2));
snr_gain_tukey = std(sCi_tukey(:))/std(sCi_standard(:)) % gain observed in the Ci based Tukey's fence relative to the Ci based on original accuracy


%% shows high z-score values of the smooth group classification images with a face outline background to help with interpretation
maxi = max([max(sCi_prop(:)), max(sCi_standard(:)), max(sCi_reclass(:)), max(sCi_tukey(:)), max(sCi_wise_and_ma(:)), max(sCi_reject(:))]); % maximum of all Ci's
mini = 8; % minimum displayed z-score

the_colormap = inferno; % perceptually uniform colormap
the_colormap = the_colormap(end:-1:1,:); % inverses colormap
the_colormap(1,:) = [1 1 1]; % white

back = double(imread('background.tif'))/255; % the face outline backgrround
%back = ones(xySize); % white background

% classification image based on original accuracy
temp2 = sCi_standard(:); % vectorizes Ci
temp2 = (temp2-mini)/(maxi-mini); % stretches between mini and maxi
temp2 = max(temp2, 0); % makes sure that the min is 0
temp2 = ceil(temp2 * 255) + 1; % makes the vectorized Ci vary between 1 and 255
temp = the_colormap(temp2,:); % applies the colormap
sCi_col = imresize(reshape(temp, [size(sCi) 3]), size(back,1)/size(sCi,1)); % reshapes in 2D and resize to match background size
im = sCi_col.*back + (1-back)/2; % combines with background
figure, imshow(im) % shows high z-score values in the group Ci over a face outline to help with interpretation.
title('Original classification') % adds a title

% classification image based on the optimal reclassification procedure
temp2 = sCi_reclass(:);
temp2 = (temp2-mini)/(maxi-mini); % stretches between mini and maxi
temp2 = max(temp2, 0);
temp2 = ceil(temp2 * 255) + 1;
temp = the_colormap(temp2,:);
sCi_col = imresize(reshape(temp, [size(sCi) 3]), size(back,1)/size(sCi,1));
im = sCi_col.*back + (1-back)/2;
figure, imshow(im)
title('Reclassification')

% classification image based on the unique mean percentage of correct responses reclassified mean percent of reclassification
temp2 = sCi_prop(:);
temp2 = (temp2-mini)/(maxi-mini); % stretches between mini and maxi
temp2 = max(temp2, 0);
temp2 = ceil(temp2 * 255) + 1;
temp = the_colormap(temp2,:);
sCi_col = imresize(reshape(temp, [size(sCi) 3]), size(back,1)/size(sCi,1));
im = sCi_col.*back + (1-back)/2;
figure, imshow(im)
title('Mean reclassification %')

% classification image based on the Wise & Ma (2012) criterion that is, > 90% of mean
temp2 = sCi_wise_and_ma(:);
temp2 = (temp2-mini)/(maxi-mini); % stretches between mini and maxi
temp2 = max(temp2, 0);
temp2 = ceil(temp2 * 255) + 1;
temp = the_colormap(temp2,:);
sCi_col = imresize(reshape(temp, [size(sCi) 3]), size(back,1)/size(sCi,1));
im = sCi_col.*back + (1-back)/2;
figure, imshow(im)
title('Wise & Ma')

% classification image based on on the the mean RT + 2 RT standard deviation criterion
temp2 = sCi_reject(:);
temp2 = (temp2-mini)/(maxi-mini); % stretches between mini and maxi
temp2 = max(temp2, 0);
temp2 = ceil(temp2 * 255) + 1;
temp = the_colormap(temp2,:);
sCi_col = imresize(reshape(temp, [size(sCi) 3]), size(back,1)/size(sCi,1));
im = sCi_col.*back + (1-back)/2;
figure, imshow(im)
title('Mean + 2 st. dev.')

% classification image based on Tukey's fence (Tukey, 1977)
temp2 = sCi_tukey(:);
temp2 = (temp2-mini)/(maxi-mini); % stretches between mini and maxi
temp2 = max(temp2, 0);
temp2 = ceil(temp2 * 255) + 1;
temp = the_colormap(temp2,:);
sCi_col = imresize(reshape(temp, [size(sCi) 3]), size(back,1)/size(sCi,1));
im = sCi_col.*back + (1-back)/2;
figure, imshow(im)
title('Tukey')



