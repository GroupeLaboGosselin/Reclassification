% Test of reclassification procedure with the data of Faghel-Subeyrand et al. (2019)

% load data
load('faghel_soubeyrand2019.mat');
nb_blocks = 3;
nb_subjects = 140;
xySize = 128; % Ci width and height in pixels

% initialize CIs
Ci_vect_reclass = zeros(nb_subjects, xySize^2);
Ci_vect_standard = zeros(nb_subjects, xySize^2);
Ci_vect_RT = zeros(nb_subjects, xySize^2);
Ci_vect_RT_bins = zeros(nb_subjects, xySize^2);

% main loop
for subject = 1:nb_subjects,
    
    subject

    % prepare data
    tmp = squeeze(double(fred.acc(subject,:,:)));
    tmprt = squeeze(double(fred.rts(subject,1:nb_blocks,:)));
    
    RT = zeros(1,nb_blocks*100);
    accuracy = zeros(1,nb_blocks*100);

    for ii = 1:nb_blocks,
        accuracy(1,(ii-1)*100+1:ii*100) = tmp(ii,:);
        RT(1,(ii-1)*100+1:ii*100) = tmprt(ii,:);
    end
 
    mean_accuracies(subject) = mean(accuracy);
    mean_RT(subject) = mean(RT);
    median_RT(subject) = median(RT);
    mean_RT_correct(subject) = mean(RT(find(accuracy==1)));
    mean_RT_incorrect(subject) = mean(RT(find(accuracy==0)));
    
    tmp = squeeze(double(fred.bub_mask(subject,:,:,:)));    % 100(subjects) x 6(blocks) x 100(trials) x 16384(pixels)
    X = zeros(nb_blocks*100, 16384);
    
    for ii = 1:nb_blocks,
        X((ii-1)*100+1:ii*100,:) = tmp(ii,:,:);
    end
    std_tmp = std(X,0,2);
    
    X = (X - repmat(mean(X,2), 1, 128^2)) ./ repmat(std_tmp, 1, 128^2); % normalized X
    X(isnan(X)) = 0; % remove nan's due to 0 bubbles trials

    % make all Ci's

    % accuracy Ci
    y_standard = (accuracy-mean(accuracy))/std(accuracy); % normalized y
    Ci_vect_standard(subject,:) = y_standard*X/sqrt(length(accuracy));
    
    % RT Ci   
    y_RT = -(RT-mean(RT))/std(RT); % normalized y
    Ci_vect_RT(subject,:) = y_RT*X/sqrt(length(accuracy));
       
    % RT bins Ci
    [rRT, ii] = sort(RT);   % in ascending order (fastest RTs first)
    temp = max(ii):-1:1;    % to give positive weight to fast RTs
    ranks(ii) = temp;
    n = max(ranks); % nb of bins; when nb of bins = max this is equivalent to the rank
    bins = ceil(ranks*n/max(ranks)); % reordered 
    y_bins = (bins-mean(bins))/std(bins); % normalized y
    Ci_vect_RT_bins(subject,:) = y_bins*X/sqrt(length(accuracy));
 
    % accuracy reclassified Ci
    [accuracy_reclass stats] = reclassify(accuracy, RT); 
    y_reclass = (accuracy_reclass - mean(accuracy_reclass)) / std(accuracy_reclass);
    Ci_vect_reclass(subject,:) = y_reclass*X/sqrt(length(accuracy));

end

% smooth Ci
sigma = 5;
TNoyau = 6*sigma;
gauss = fspecial('gaussian',ceil(TNoyau),sigma);

Ci = reshape(squeeze(sum(Ci_vect_standard, 1)), xySize, xySize)/sqrt(nb_subjects);
sCi = filter2(gauss, Ci);
sCi_standard = sCi/sqrt(sum(gauss(:).^2));

Ci = reshape(squeeze(sum(Ci_vect_RT, 1)), xySize, xySize)/sqrt(nb_subjects);
sCi = filter2(gauss, Ci);
sCi_RT = sCi/sqrt(sum(gauss(:).^2));

Ci = reshape(squeeze(sum(Ci_vect_RT_bins, 1)), xySize, xySize)/sqrt(nb_subjects);
sCi = filter2(gauss, Ci);
sCi_RT_bins = sCi/sqrt(sum(gauss(:).^2));

Ci = reshape(squeeze(sum(Ci_vect_reclass, 1)), xySize, xySize)/sqrt(nb_subjects);
sCi = filter2(gauss, Ci);
sCi_reclass = sCi/sqrt(sum(gauss(:).^2)); % normalized for Gaussian kernel


% make nice figures
maxi = max([max(sCi_standard(:)), max(sCi_reclass(:)), max(sCi_RT(:)), max(sCi_RT_bins(:))]);
mini = 8; % minimum displayed z-score

the_colormap = hot;
the_colormap = the_colormap(end:-1:1,:); % inverse colormap
the_colormap(1,:) = [1 1 1]; % white

back = double(imread('background.tif'))/255;
clrmap_len=(size(the_colormap, 1)-1);
% standard
temp2 = sCi_standard(:);
temp2 = (temp2-mini)/(maxi-mini); % stretches between mini and maxi
temp2 = max(temp2, 0);
temp2 = ceil(temp2 * clrmap_len) + 1;
temp = the_colormap(temp2,:);
sCi_col = imresize(reshape(temp, [size(sCi) 3]), size(back,1)/size(sCi,1));
im = sCi_col.*back + (1-back)/2;
figure, imshow(im)
title('standard')

% RT
temp2 = sCi_RT(:);
temp2 = (temp2-mini)/(maxi-mini); % stretches between mini and maxi
temp2 = max(temp2, 0);
temp2 = ceil(temp2 * clrmap_len) + 1;
temp = the_colormap(temp2,:);
sCi_col = imresize(reshape(temp, [size(sCi) 3]), size(back,1)/size(sCi,1));
im = sCi_col.*back + (1-back)/2;
figure, imshow(im)
title('RT')

% RT bins
temp2 = sCi_RT_bins(:);
temp2 = (temp2-mini)/(maxi-mini); % stretches between mini and maxi
temp2 = max(temp2, 0);
temp2 = ceil(temp2 * clrmap_len) + 1;
temp = the_colormap(temp2,:);
sCi_col = imresize(reshape(temp, [size(sCi) 3]), size(back,1)/size(sCi,1));
im = sCi_col.*back + (1-back)/2;
figure, imshow(im)
title('RT bins')

% reclass
temp2 = sCi_reclass(:);
temp2 = (temp2-mini)/(maxi-mini); % stretches between mini and maxi
temp2 = max(temp2, 0);
temp2 = ceil(temp2 * clrmap_len) + 1;
temp = the_colormap(temp2,:);
sCi_col = imresize(reshape(temp, [size(sCi) 3]), size(back,1)/size(sCi,1));
im = sCi_col.*back + (1-back)/2;
figure, imshow(im)
title('reclass')


