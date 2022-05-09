
% sex discrimination task
sujets = {
    {'Sara', [1:2], 1};
    {'CV', [1:2], 1};
    {'SC', [1:2], 1};
    {'MPL', [1:2], 1};
    {'SJ', [1:2], 1};
    {'SS', [1:2], 1};
    {'SRA', [1:2], 1};
    {'ABG', [1:2], 1};
    {'CDT', [1:2], 1};
    {'GBP', [1:2], 1};
    {'MMG', [2], 1}; % just one valid session
    {'JM', [1:2], 1};
    };

toi = 177:380; 

stdTime = 1.8;
nfeatures = 3;
nframes = 24;
crit0 = 0.1; % in s
crit1 = 2; % in s
crit2 = 3; % in std dev, for behavior
crit3 = 3; % in std dev, for EEG

nperms_behav = 500;
nperms = 500;

TNoyau = ceil(6*stdTime);
bx = [0:TNoyau]-TNoyau/2;
bubble = exp(-(bx/stdTime).^2);
bubble = bubble/max(bubble(:));

reclass_crit = NaN(length(sujets),2);
reclass_nb = NaN(length(sujets),2);
reclass_eff = NaN(length(sujets),2);
reclass_gain = NaN(length(sujets),2);
mean_RT = NaN(length(sujets),2);
corr_RT = NaN(length(sujets),2);
incorr_RT = NaN(length(sujets),2);
mean_acc = NaN(length(sujets),2);
ntrials = NaN(length(sujets),2);

ERP_orig = single(zeros(length(sujets),2,62,length(toi)));
ERP_reclass = single(zeros(length(sujets),2,62,length(toi)));
ERP_wise = single(zeros(length(sujets),2,62,length(toi)));
ERP_ext = single(zeros(length(sujets),2,62,length(toi)));
ERP_tukey = single(zeros(length(sujets),2,62,length(toi)));
ERP_mrt = single(zeros(length(sujets),2,62,length(toi)));
for sujet = 1:length(sujets)
    
    disp(sujet)
    
    for session = sujets{sujet}{2}

        % collect data from each session/block
        load(sprintf('scd_data_case1_spline_%s_%i.mat', sujets{sujet}{1}, session)) %%%%%
        eegdata = scd_data_case1;
        data = cell(1,4);
        for ii = 1:4
            data{ii} = load(sprintf('/Volumes/DATA/ACADEMIA/MEG_dynamic/EEG_results/cmpt/facefeattime_%i_%s_%i.mat', sujets{sujet}{3}, sujets{sujet}{1}, (session-1)*4+ii));
            data{ii}.Xpadded = data{ii}.Xpadded;
            data{ii}.correct = data{ii}.correct;
        end
        prop_all = [data{1}.prop, data{2}.prop, data{3}.prop, data{4}.prop];
        correct_all = [zscore(data{1}.correct), zscore(data{2}.correct), zscore(data{3}.correct), zscore(data{4}.correct)];
        correct_01 = [data{1}.correct, data{2}.correct, data{3}.correct, data{4}.correct];
        X_all = cat(1, data{1}.Xpadded, data{2}.Xpadded, data{3}.Xpadded, data{4}.Xpadded);
        RT_01 = [data{1}.RT, data{2}.RT, data{3}.RT, data{4}.RT];
        RT_all = [zscore(data{1}.RT), zscore(data{2}.RT), zscore(data{3}.RT), zscore(data{4}.RT)];

        % EEG activity for each electrode, trial, and time point
        y = zeros(62,length(eegdata.trial),length(toi));
        parfor elec = 1:62
            y_temp = zeros(length(eegdata.trial),length(toi));
            for trial = 1:length(eegdata.trial)
                y_temp(trial,:) = eegdata.trial{trial}(elec,toi);
            end
            y(elec,:,:) = y_temp;
        end
        yR = y;
        
        % do not use invalid trials
        seltrials = true(1,length(eegdata.trial));
        seltrials(trials_to_reject) = false;
        if strcmp(sujets{sujet}{1},'SB') && session==1
            seltrials = seltrials(276:end);
            y = y(:,276:end,:);
            yR = yR(:,276:end,:);
        end
        seltrials(RT_01<crit0 | RT_01>crit1 | RT_all>crit2 | RT_all<-crit2) = false;
        seltrials(prop_all==1) = false;      
        correct_01 = correct_01(seltrials);
        RT_01 = RT_01(seltrials);
        ysel = single(yR(:,seltrials,:));
        
        % reclassification
        [correct_reclass, stats] = reclassify(correct_01,RT_01);
        
        % diverse statistics
        reclass_crit(sujet,session) = stats.reclass_criterion;
        reclass_nb(sujet,session) = length(stats.reclass_index);
        ntrials(sujet,session) = length(RT_01);
        reclass_eff(sujet,session) = stats.reclass_efficiency;
        reclass_gain(sujet,session) = stats.reclass_gain;
        mean_acc(sujet,session) = mean(correct_01);
        mean_RT(sujet,session) = mean(RT_01);
        corr_RT(sujet,session) = mean(RT_01(correct_01==1));
        incorr_RT(sujet,session) = mean(RT_01(correct_01==0));

        % reclassify using mean reclassification threshold
        correct_meanreclass = correct_01;
        thresh = quantile(RT_01,1-0.0766);
        correct_meanreclass(RT_01>thresh) = 0;

        % using method from Wise & Ma
        correct_wise = correct_01;
        correct_wise(correct_01==1 & RT_01>(mean(RT_01)*1.9)) = 0;
        
        % using mean + 2 std method
        correct_ext = correct_01;
        correct_ext(RT_01>(mean(RT_01)+2*std(RT_01))) = 0;
        
        % using Tukey's Fences method
        q1 = quantile(RT_01,.25);
        q3 = quantile(RT_01,.75);
        iqr = q3 - q1;
        correct_tukey = correct_01;
        correct_tukey(RT_01>(q3+1.5*iqr)) = 0;
        
        % compute ERP correct/incorrect comparison using each method
        ERP_orig(sujet,session,:,:) = squeeze(mean(ysel(:,correct_01==1,:),2)) - squeeze(mean(ysel(:,correct_01==0,:),2));
        ERP_ext(sujet,session,:,:) = squeeze(mean(ysel(:,correct_ext==1,:),2)) - squeeze(mean(ysel(:,correct_ext==0,:),2));
        ERP_tukey(sujet,session,:,:) = squeeze(mean(ysel(:,correct_tukey==1,:),2)) - squeeze(mean(ysel(:,correct_tukey==0,:),2));        
        ERP_reclass(sujet,session,:,:) = squeeze(mean(ysel(:,correct_reclass==1,:),2)) - squeeze(mean(ysel(:,correct_reclass==0,:),2));
        ERP_mrt(sujet,session,:,:) = squeeze(mean(ysel(:,correct_meanreclass==1,:),2)) - squeeze(mean(ysel(:,correct_meanreclass==0,:),2));
        ERP_wise(sujet,session,:,:) = squeeze(mean(ysel(:,correct_wise==1,:),2)) - squeeze(mean(ysel(:,correct_wise==0,:),2));

    end
end

% compute t-values using ERPs from each method
[h,p,ci,stats] = ttest(squeeze(mean(ERP_orig,2)));
[h,p,ci,stats_reclass] = ttest(squeeze(mean(ERP_reclass,2)));
[h,p,ci,stats_ext] = ttest(squeeze(mean(ERP_ext,2)));
[h,p,ci,stats_tukey] = ttest(squeeze(mean(ERP_tukey,2)));
[h,p,ci,stats_wise] = ttest(squeeze(mean(ERP_wise,2)));
[h,p,ci,stats_mrt] = ttest(squeeze(mean(ERP_mrt,2)));
