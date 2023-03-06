import numpy as np
import scipy.interpolate as interp
import scipy.stats as stats

def reclassify(accuracy, reclass_evidence):
    """
    The function takes 2 inputs: _accuracy_, a vector equal to 0 when the response to a trial was incorrect and to 1 when it 
    was correct; and _reclass_evidence_, a vector of the same length as _accuracy_ equal to a reclassification evidence for each 
    corresponding trial such as the response times. Note that the average of the evidence for incorrect responses ca be either 
    greater or smaller than the average of the evidence for correct responses but it must be different. The The function has 2 
    outputs: _accuracy_reclass_, a vector of the same length as _accuracy_ equal to 0 when the response to a trial was incorrect 
    or when a correct response to a trial was reclassified as incorrect, and to 1 when the response to trial was correct and 
    wasn't reclassified as incorrect; and _stats_reclass_ a structure with 5 fields: _reclass_evidence_criterion_, a reclassification 
    evidence such that when _reclass_evidence_polarity_ * _reclass_evidence_ > _reclass_evidence_polarity_ * _reclass_evidence_criterion_ 
    a correct response was reclassified as incorrect; _reclass_evidence_polarity_, either 1 or -1 and indicating how to interpret 
    the criterion; _reclass_index_, an index of the correct responses reclassified as incorrect; _reclass_efficiency_, the estimated 
    proportion of true correct and incorrect responses minus false correct and incorrect responses following reclassification; 
    _reclass_gain_, the ratio between _reclass_efficiency_ and the efficiency prior to reclassification (note that sqrt(_reclass_gain_) 
    provides an approximation of the expected SNR gain, assuming that all trials carry the same information); and _t_, which contains 
    statistics about a two-sample t-test on the mean of the reclassification evidence for correct and for incorrect responses (if _t.tstat_).

    Gosselin, F., Daigneault, V., Larouche, J.-M. & Caplette, L. (submitted). Reclassifying guesses to increase signal-to-noise ratio
    in psychological experiments.

    Frederic Gosselin, 01/06/2020
    frederic.gosselin@umontreal.ca

    Adapted to Python by
    Laurent Caplette, 17/08/2020
    laurent.caplette@yale.edu
    """

    reclass_evidence = np.array(reclass_evidence).astype(np.float32)  # in case a list is feeded
    accuracy = np.array(accuracy).astype(np.float32)  # in case a list is feeded

    if not all(np.unique(accuracy) == [0, 1]):  # check that accuracy is composed of only zeros and ones
        raise ("'accuracy' variable must be composed of zeros and ones")

    if not accuracy.shape == reclass_evidence.shape:  # check that variables are of the same size
        raise ("'accuracy' and reclass_evidence must have the same size")

    polarity = 1  # default evidence polarity
    reclass_evidence_incorrect = reclass_evidence[accuracy == 0]
    reclass_evidence_correct = reclass_evidence[accuracy == 1]
    t = stats_reclass.t_test_ind(reclass_evidence_correct, reclass_evidence_incorrect)
    if t.statistics < 0:  # the evidence is greater for incorrect than correct trials
        polarity = -1  # change evidence polarity
    reclass_evidence *= polarity  # the evidence multiplied by its polarity

    nb_std = 2
    outliers = reclass_evidence > np.mean(reclass_evidence) + nb_std * np.std(reclass_evidence)  # temporary outliers to help frame the histogram
    _, bins = np.histogram(reclass_evidence[np.logical_not(outliers)], 'fd')  # uses the Freedman-Diaconis rule for bin width
    bin_width = bins[1] - bins[0]  # bin width
    bins = np.arange(bins[0], np.ceil(np.amax(reclass_evidence) / bin_width) * bin_width, bin_width)  # complete evidence range, including outliers
    correct_evidence = reclass_evidence[accuracy == 1]  # correct response evidences
    n_correct, _ = np.histogram(correct_evidence, bins)  # correct evidences histograms
    incorrect_evidence = reclass_evidence[accuracy == 0]  # incorrect response evidences
    n_incorrect, _ = np.histogram(incorrect_evidence, bins)  # incorrect evidences histogram; this is also the false correct evidences histogram

    # calculates frequency distribution
    x = (bins[:-1] + bins[1:]) / 2  # centers of the histogram bins
    s_x = np.linspace(np.amin(x), np.amax(x), ((np.amax(x) - np.amin(x)) // .01).astype(np.int32))  # fine histogram bins for interpolation
    for ii in range(len(bins) - 1):  # replaces histogram bin centers by histogram bin averages whenever possible
        ind = np.where(np.logical_and(reclass_evidence >= bins[ii], reclass_evidence < bins[ii + 1]))[0]
        if ind.size != 0:
            x[ii] = np.mean(reclass_evidence[ind])
    f1 = interp.CubicSpline(x, n_correct)
    s_n_correct = f1(s_x)
    f2 = interp.CubicSpline(x, n_incorrect)
    s_n_incorrect = f2(s_x)
    s_n_true_correct = s_n_correct - s_n_incorrect

    # finds the best evidence criteria
    N = np.sum(s_n_correct) + np.sum(s_n_incorrect)  # number of points in all interpolated frequency distributions; general case
    I_o = np.sum(s_n_incorrect)  # number of points in interpolated n_incorrect frequency distribution
    cCR = np.cumsum(s_n_true_correct)  # cumulative interpolated true correct evidence frequency distribution
    cM = np.cumsum(s_n_incorrect)  # cumulative interpolated false correct evidence frequency distribution
    s_efficiency = (4 * I_o - N + 2 * (cCR - cM)) / N  # interpolated efficiency as a function of evidence reclassification criterion; general case
    s_ind = np.argmax(s_efficiency)
    reclass_criterion = s_x[s_ind]  # chosen evidence criterion

    # reclassifies correct responses as incorrect responses
    accuracy_reclass = accuracy  # initialize with old accuracy
    reclass_index = np.where(np.logical_and(accuracy == 1, (reclass_evidence > reclass_criterion)))[0]  # which correct response should be reclassified as an incorrect
    accuracy_reclass[reclass_index] = 0

    # some statistics
    stats_reclass = {
        'reclass_t': t,
        'reclass_polarity': polarity,
        'reclass_criterion': polarity * reclass_criterion,
        'reclass_index': reclass_index,
        'reclass_efficiency': s_efficiency[s_ind],
        #'reclass_gain': s_efficiency[s_ind] / s_efficiency[-1]
        'reclass_gain': s_efficiency[s_ind] / (1-2*(1-np.mean(accuracy)))
    }

    return accuracy_reclass, stats_reclass