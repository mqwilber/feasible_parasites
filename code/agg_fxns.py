from __future__ import division
import numpy as np
import pandas as pd
import macroeco.models as mod
import macroeco.compare as comp
from sklearn import tree
import pydot
from sklearn.feature_extraction import DictVectorizer
import os
import scipy.stats as stats
gaussian_kde = stats.gaussian_kde
import matplotlib.pyplot as plt
from scipy.special import gammaln

# Check for pypartitions
try:
    import pypartitions as pyp # Package obtained from https://github.com/klocey/partitions
except ImportError:
    print("Feasible set package not found. Download from\n" +
                "https://github.com/klocey/partitions and set to PYTHONPATH")

# A lot of folks won't have rpy2
try:

    import rpy2.robjects as robjects
    import pandas.rpy.common as com
    R = robjects.r

    try:
        robjects.r("library(Matching)")
        robjects.r("library(kSamples)")
    except Exception:
        print("Please install.packages the necessary R packages")

except ImportError:
    print("Rpy2 package not found. Functions that rely on R will not work")


"""

Description
------------
Analysis functions for examining how top-down null models describe the
distribution of parasites across hosts.  Used in the manuscript
feasible_manuscript.tex.


Author
------

Mark Wilber, UCSB, 2016

"""

def feasible_mixture(para_host_vec, samples=200, center="mean"):
    """
    Gives a feasible set mixture from a given parasite host vector.

    Parameters
    -----------
    para_host_vec : list of tuples
        Length of the list is the number of heterogeneities. Each tuple is
        (P, H) where P is parasites and H is hosts. e.g. [(100, 10), (20, 30)]

    samples : int
        Number of feasible set samples to take

    center : str
        Either "mean", "median", or "mode".  They give very similar answers.  The mean
        will guarantee that the mean of the returned predicted vector will
        equal the expected mean for all sample sizes.  This is not necessarily
        True for the median or mode for small sample sizes. For the mode, if
        multiple values are found for the mode the minimum value is taken. The
        mode measure is more dependent on sample size than the mean or median,
        though it is what is used in Locey and White 2013.

    Returns
    -------
    : 2D array, 1D array
        The sorted array of all the sampled feasible sets
        The predicted center of the feasible set ("mean", "median", "mode")

    Examples
    --------
    full_feas, cent_feas = feasible_mixture([(100, 10), (20, 30)],
                                samples=200,  center="median")

    """

    mixed_sample = []
    para_host_vec = convert_to_int(para_host_vec)

    for ph in para_host_vec:

        if ph[0] == 0: # Hosts don't have any parasites

            tfeas = np.zeros(ph[1] * samples).reshape(samples, ph[1])
            mixed_sample.append(tfeas)

        else:

            tfeas = pyp.rand_partitions(int(ph[0]), int(ph[1]), samples, zeros=True)
            mixed_sample.append(tfeas)

    mix_feas = np.concatenate(mixed_sample, axis=1)
    sorted_feas = np.sort(mix_feas)[:, ::-1]

    if center == "mean":
        med_feas = np.mean(sorted_feas, axis=0)
    elif center == "median":
        med_feas = np.median(sorted_feas, axis=0)
    else:
        med_feas = stats.mode(sorted_feas, axis=0)[0].flatten()

    return sorted_feas, med_feas


def nbd_mixture(para_host_vec, k=1, finite=True):
    """
    Returns the rank abundance distribution predicted from a negative
    binomial distribution.

    When k=1, the prediction is equivalent to a Max Ent prediction with a
    constraint on the mean.

    When finite=True, the prediction is from a finite negative binomial

    When finite=False, the prediction is from a negative binomial.

    Parameters
    -----------

    para_host_vec : list of tuples
        Length of the list is the number of heterogeneties. Each tuple is
        (P, H) where P is parasites and H is hosts

    k : float
        Aggregation parameter of the negative binomial

    finite : bool
        If true uses a finite negative binomials (Zillio and He 2011).
        Other wises uses a pure negative binomial

    Returns
    -------
    : array
        A rank-abundance vector sorted from highest abundance to lowest
        abundance as predicted by the negative binomial

    Examples
    --------

    # Maxent prediction
    nbd_pred = nbd_mixture([(20, 20), (30, 10)], k=1, finite=True)
    """

    mixed_sample = []
    para_host_vec = convert_to_int(para_host_vec)

    for ph in para_host_vec:

        if ph[0] == 0: # Hosts have zero parasites
            mixed_sample.append(list(np.zeros(ph[1])))

        elif ph[1] == 1: # Only one host
            mixed_sample.append([ph[0]])

        else:

            if finite:
                mixed_sample.append(mod.cnbinom.rank(ph[1], ph[0] /
                                            ph[1], k, ph[0]))
            else:
                mixed_sample.append(mod.nbinom.rank(ph[1], ph[0] /
                                            ph[1], k))

    return np.sort(np.concatenate(mixed_sample))[::-1]


def poisson_mixture(para_host_vec, finite=True):
    """
    Returns the rank distribution predicted by a Poisson mixture

    If finite == True, than a binomial distribution is used instead of a
    Poisson.

    Parameters
    -----------

    para_host_vec : list of tuples
        Length of the list is the number of heterogeneties. Each tuple is
        (P, H) where P is parasites and H is hosts

    finite : bool
        If True, uses a binomial distribution.

    Returns
    -------
    : array
        Rank abundance distribution for the given mixture

    Examples
    --------

    # Poisson prediction
    pois_pred = poisson_mixture([(20, 20), (30, 10)], finite=False)

    """

    mixed_sample = []
    para_host_vec = convert_to_int(para_host_vec)

    def poisson_rank(n, mu):
        return stats.poisson.ppf((np.arange(1, n + 1) - 0.5) / n, mu)

    def binomial_rank(n, P):
        return stats.binom.ppf((np.arange(1, n + 1) - 0.5) / n, P, 1 / n)

    for ph in para_host_vec:

        if ph[0] == 0: # Hosts have zero parasites
            mixed_sample.append(np.zeros(ph[1]))

        elif ph[1] == 1: # Only one
            mixed_sample.append([ph[0]])

        else:
            if finite:
                mixed_sample.append(binomial_rank(ph[1], ph[0]))

            else:
                mixed_sample.append(poisson_rank(ph[1], ph[0] / ph[1]))

    return np.sort(np.concatenate(mixed_sample))[::-1]


def weighted_feasible_mixture(para_host_vec, samples=200, center="median",
                                        model="composition"):
    """
    """

    if model == "partition":
        sorted_feas, pred = feasible_mixture(para_host_vec, samples=samples,
                                                center=center)
    elif model == "composition":

        mixed_sample = []
        para_host_vec = convert_to_int(para_host_vec)

        for ph in para_host_vec:

            if ph[0] == 0:  # Hosts don't have any parasites

                tfeas = np.zeros(ph[1] * samples).reshape(samples, ph[1])
                mixed_sample.append(tfeas)

            else:

                tfeas_mh, _ = constrained_ordered_set(ph[0], ph[1],
                                maxent_weight_fxn,
                                feasible_proposal, feasible_ratio,
                                samples=samples*2,
                                sampling="metropolis-hastings")
                # Discard the first half as burn in
                post_samps = tfeas_mh[(samples + 1):, :]
                mixed_sample.append(post_samps)

        mix_feas = np.concatenate(mixed_sample, axis=1)
        sorted_feas = np.sort(mix_feas)[:, ::-1]

    elif model == "multinomial":

        mixed_sample = []
        para_host_vec = convert_to_int(para_host_vec)

        for ph in para_host_vec:

            if ph[0] == 0: # Hosts don't have any parasites

                tfeas = np.zeros(ph[1] * samples).reshape(samples, ph[1])
                mixed_sample.append(tfeas)

            else:

                tfeas = np.random.multinomial(ph[0], np.repeat(1 / ph[1], ph[1]), 
                                            size=samples)
                mixed_sample.append(tfeas)

        mix_feas = np.concatenate(mixed_sample, axis=1)
        sorted_feas = np.sort(mix_feas)[:, ::-1]

    else:
        raise TypeError("{0} not a recognized model".format(model))

    if center == "mean":
        med_feas = np.mean(sorted_feas, axis=0)
    elif center == "median":
        med_feas = np.median(sorted_feas, axis=0)
    else:
        med_feas = stats.mode(sorted_feas, axis=0)[0].flatten()

    return sorted_feas, med_feas


def cart_analysis(para_data, y_name, x_names, print_tree=False,
        max_leaf_nodes=None, use_R=False, prune=True, cp=0.0001, minsplit=2,
        min_samples_leaf=2, filename="output.dot"):
    """
    Function takes in parasite data and returns the (P, H) values from the
    terminal nodes of the regression tree. The output is in the form
    [(P_1, H_1), (P_2, H_2), ..., (P_n, H_n)] and can be used directly in the
    mixture models given above.

    The CART analysis can either be done in R or Python. In R you have the
    ability to prune the tree to get the optimal tree.  This is a random
    procedure so you will get different trees each time. Python CART does not
    have the pruning feature.

    Parameters
    ----------
    para_data : DataFrame
        Parasite data with covariates.  Must be a DataFrame
    y_name : str
        Name of the response variable
    x_names : list of strs
        list of strings that are the names of the predictor variables in
        para_data. They must be column names in para_data
    max_leaf_nodes : int
        The number of terminal leaf nodes to include in the regression tree
    print_tree: bool
        If True, prints and saves the regression tree
    use_R : bool
        If True, uses rpart for the regression tree analysis
    prune : bool
        Only valid if use_R is True.  If True, prunes the tree
    cp : float
        Only used for rpart. A lower value allows for a larger tree to be fit
    minsplit : int
        Minimum number of observations a node needs to have in order to be split
    min_samples_leaf : int
        The minimum number of samples a leaf node must contain
    filename : str
        The filename that the dot files (the tree representation) is saved to
        if print_tree is True.

    Returns
    -------
    : 1. list of tuples
        Parasite-host vectors
      2. None in use_R == True, else a fitted python TREE object

    Examples
    --------
    See manuscript_analysis_discrete_known_heterogeneity.py for examples of
    using this function.


    """
    if use_R:

        # R regression tree with pruning

        R("""library(rpart)""")

        robjects.globalenv['temp_data'] = com.convert_to_r_dataframe(para_data)
        robjects.globalenv['x_names'] = robjects.StrVector(x_names)
        robjects.globalenv['y_name'] = robjects.StrVector([y_name])
        robjects.globalenv['prune'] = robjects.BoolVector([prune])
        robjects.globalenv['cp'] = robjects.FloatVector([cp])
        robjects.globalenv['minsplit'] = robjects.FloatVector([minsplit])

        # Run R analysis
        R(rcart_analysis)

        # Extract data from R
        results = com.convert_robj(robjects.globalenv['means'])
        num_in_groups = com.convert_robj(robjects.globalenv['num_in_groups'])

        # Rename and convert
        hosts, parasites = convert_R_to_python(results, num_in_groups)

    else:

        # Python implementation...no pruning

        Y = np.array(para_data[y_name])

        x_data = np.array(para_data[x_names])
        xs = [{name: val for name, val in zip(x_names, vals)} for vals in x_data]
        vec = DictVectorizer()
        X = vec.fit_transform(xs).toarray()

        clf = tree.DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,
                                    min_samples_split=minsplit,
                                    min_samples_leaf=min_samples_leaf)
        fit = clf.fit(X, Y)

        term_ind = fit.tree_.feature == -2
        means = np.resize(fit.tree_.value, (fit.tree_.node_count,))[term_ind]
        hosts = fit.tree_.n_node_samples[term_ind]
        parasites = np.round(hosts * means, decimals=0)

    if print_tree:

        if use_R:

            robjects.globalenv['filename'] = robjects.StrVector([filename])
            R(print_r_tree)
        else:

            if os.path.isfile(filename):
                os.remove(filename)

            with open(filename, 'w') as fout:
                tree.export_graphviz(clf, feature_names=vec.get_feature_names(),
                    out_file=fout)

        # Open the dot file with "dot -Tpng output.dot > temp.png" in BASH
    if use_R:
        return zip(parasites, hosts), None
    else:
        return zip(parasites, hosts), fit

def convert_R_to_python(results, num_in_groups):
    """ Helper function to convert R results to python """

    results = results.rename(columns={'X2':"group_id", "X1": "means"})
    results.sort(columns="group_id", inplace=True)
    num_in_groups.sort(columns="Var1", inplace=True)
    results['hosts'] = num_in_groups['Freq']
    results['parasites'] = results.hosts * results.means

    return (np.round(results.hosts, decimals=0),
            np.round(results.parasites, decimals=0))

rcart_analysis = """

            # Make the formulas

            form_str = ""
            for(i in 1:length(x_names)){

                if(i == 1){
                    form_str = paste(form_str, x_names[i], sep="")
                } else {
                    form_str = paste(form_str, x_names[i], sep="+")
                }
            }

            form_str = paste(y_name, form_str, sep="~")

            fit1 = rpart(formula(form_str), data=temp_data, cp=cp,
                        minsplit=minsplit)

            cptable = fit1$cptable
            low_cp = cptable[which.min(cptable[, "xerror"]), 1]

            if(prune){
                fit2 = prune.rpart(fit1, low_cp)
            } else {fit2 = fit1}

            td = data.frame(cbind(predict(fit2), fit2$where))
            means = aggregate(X1 ~ X2, data=td, mean)
            num_in_groups = data.frame(table(fit2$where))

        """

print_r_tree = """
                library(rattle)
                pdf(filename)
                fancyRpartPlot(fit2)
                dev.off()

                # Save variable importance
                write.csv(data.frame(fit2$variable.importance),
                            paste(filename, "variable_importance", sep=""))

            """

def get_feasible_predictions_from_data(para_data, para_name, initial_split,
                samples=200, para_min=20, host_min=6, output=True,
                center="mean", logger=None, large_samples=200,
                fs_type="partition"):
    """
    For each unique group given by initial_split (Host ID,  Site, etc),
    function calculates the feasible set prediction from the data. Returns
    a dictionary where the keys are the unique identifiers from the column
    initial_split and they look up a tuple that contains three items:

    1. The full feasible set prediction
    2. The mean/median/mode feasible set prediction
    3. The observed data

    Parameters
    -----------
    para_data: DataFrame
        Pandas dataframe with parasite data
    para_name : str
        Name of the parasite column in the data set
    initial_split : str
        Name of the column in the dataset where the initial split will occur
    samples : int
        Number of feasible set samples
    para_min : int
        The minimum number of parasites an entity (Site or host) must contain
        to be included in the analysis
    output : bool
        If True, a progress message is printed after each iteration of the for
        loop.
    center : str
        Either "mean", "median", or "mode". Specifies what center of the feasible set
        to return
    logger: tuple or None
        Tuple of logger, int. int is printed in the logging. This analysis can
        take awhile so it is helpful to log the results.
    fs_type: str
        Specifies which type of feasible set to sample. The options are
            `partition`: Unlabeled hosts and unlabeled parasites. Macrostates
                         are equally weighted (e.g. Locey and White 2013).
                         This is the default
            `composition`: Labeled hosts and unlabeled parasites 
                          (maximum entropy model)
            `multinomial`: Labeled hosts and labeled parasites

    Returns
    -------
    : dict
        See above for description

    """

    # Get unique values in the initial split column.  Often will be site
    sites = para_data[initial_split].unique()

    feas_predictions = {}

    for site in sites:

        site_data = para_data[para_data[initial_split] == site]

        if len(site_data) >= host_min:

            if site_data[para_name].sum() >= para_min:

                if output:
                    print("Getting feasible set for {0} for {1}".format(site, para_name))

                if logger != None:
                    logger[0].info("Year {0}: Getting feasible set for {1} for {2}".format(logger[1], site, para_name))

                # Feasible set predictions
                P = site_data[para_name].sum()
                H = len(site_data)

                if P < 1000: # If P too big use less samples!
                    use_samps = samples
                else:
                    use_samps = large_samples

                if fs_type == "partition":

                    full, pred = feasible_mixture([(P, H)], samples=use_samps,
                                center=center)

                elif fs_type == "composition":

                    # Could speed this up with exact composition algorithm
                    mh_me, mh_rj = constrained_ordered_set(int(P), int(H),
                                        maxent_weight_fxn,
                                        feasible_proposal, feasible_ratio,
                                        samples=2*use_samps,
                                        sampling="metropolis-hastings")

                    full = mh_me[-use_samps:, :]
                    # full = np.sort(mod.cnbinom.rvs(P / H, 1, P, size=(use_samps, H)))[:, ::-1]

                    pred = np.median(full, axis=0)

                elif fs_type == "multinomial":

                    mult_samps = np.random.multinomial(P, np.repeat(1 / H, H), 
                                                size=use_samps)
                    full = np.sort(mult_samps, axis=1)[:,::-1]
                    pred = np.median(full, axis=0)

                else:
                    raise ValueError("{0} not recognized".format(fs_type))

                feas_predictions[site] = (full, pred,
                            np.sort(np.array(site_data[para_name]))[::-1])

    return feas_predictions


def get_single_host_predictions_from_data(para_data, para_name, initial_split,
                samples=200, para_min=20, host_min=6, output=True,
                center="mean", logger=None, large_samples=200,
                fs_type="partition"):
    """
    For each unique group given by initial_split (Host ID,  Site, etc),
    function calculates the feasible set prediction from the data. Returns
    a dictionary where the keys are the unique identifiers from the column
    initial_split and they look up a tuple that contains three items:

    1. The full feasible set prediction
    2. The mean/median/mode feasible set prediction
    3. The observed data

    Parameters
    -----------
    para_data: DataFrame
        Pandas dataframe with parasite data
    para_name : str
        Name of the parasite column in the data set
    initial_split : str
        Name of the column in the dataset where the initial split will occur
    samples : int
        Number of feasible set samples
    para_min : int
        The minimum number of parasites an entity (Site or host) must contain
        to be included in the analysis
    output : bool
        If True, a progress message is printed after each iteration of the for
        loop.
    center : str
        Either "mean", "median", or "mode". Specifies what center of the feasible set
        to return
    logger: tuple or None
        Tuple of logger, int. int is printed in the logging. This analysis can
        take awhile so it is helpful to log the results.
    fs_type: str
        Specifies which type of feasible set to sample. The options are
            `partition`: Unlabeled hosts and unlabeled parasites. Macrostates
                         are equally weighted (e.g. Locey and White 2013).
                         This is the default
            `composition`: Labeled hosts and unlabeled parasites 
                          (maximum entropy model)
            `multinomial`: Labeled hosts and labeled parasites

    Returns
    -------
    : dict
        See above for description

    """

    # Get unique values in the initial split column.  Often will be site
    sites = para_data[initial_split].unique()

    feas_predictions = {}

    for site in sites:

        site_data = para_data[para_data[initial_split] == site]

        if len(site_data) >= host_min:

            if site_data[para_name].sum() >= para_min:

                if output:
                    print("Getting feasible set for {0} for {1}".format(site, para_name))

                if logger != None:
                    logger[0].info("Year {0}: Getting feasible set for {1} for {2}".format(logger[1], site, para_name))

                # Feasible set predictions
                P = site_data[para_name].sum()
                H = len(site_data)

                if P < 1000: # If P too big use less samples!
                    use_samps = samples
                else:
                    use_samps = large_samples

                if fs_type == "partition":

                    full, pred = feasible_mixture([(P, H)], samples=use_samps,
                                center=center)

                    # Approximate the pmf
                    model_pmf = feasible_pmf_approx(full)

                    #  Sample single hosts from feasible set.
                    full_flat = full.ravel()
                    fs_draws = full_flat[np.random.randint(0, len(full_flat), H * use_samps)]
                    full = fs_draws.reshape(full.shape)

                elif fs_type == "composition":

                    full = mod.cnbinom.rvs(P / H, 1, P, size=(use_samps, H))
                    model_pmf = mod.cnbinom(P / H, 1, P).pmf

                elif fs_type == "multinomial":

                    full = stats.binom.rvs(P, 1 / H, size=(use_samps, H))
                    model_pmf = stats.binom(P, 1 / H).pmf

                else:
                    raise ValueError("{0} not recognized".format(fs_type))

                feas_predictions[site] = (full, model_pmf,
                            np.sort(np.array(site_data[para_name])))

    return feas_predictions


def feasible_pmf_approx(full):
    """
    Approximate the single host pmf from a sampled feasible set

    Uses basic interpolation. Not using KDE because the distributions are
    highly right skewed.

    Parameters
    ----------
    full : 2D array
        A 2D array of samples or macrostates

    Returns
    -------
    : function
        A approximate PMF for the samples given.
    """

    # Calculate approx pmf for feasible set
    pmf = pd.value_counts(full.ravel()) / len(full.ravel())
    pmf = pmf.sort_index()

    def pmf_fxn(x):
        return(np.interp(x, pmf.index.values, pmf.values))

    return(pmf_fxn)


def get_model_predictions_from_data(para_data, para_name, initial_split,
                para_min=20, host_min=6, output=True, finite=False,
                model="top-down", heterogeneity=False):
    """
    Given a data set, gets the model predictions for each item
    in the split.  Most of the time, this will be by site.

    This function allows you to specify what type of model you are interested
    in (i.e. bottom-up or top-down), whether you want the finite of infinite
    prediction, and whether you want to include continuous heterogeneity.

    Parameters
    ----------
    para_data: DataFrame
        Pandas dataframe with parasite data
    para_name : str
        Name of the parasite column in the data set
    initial_split : str
        Name of the column in the dataset where the initial split will occur
    samples : int
        Number of feasible set samples
    para_min : int
        The minimum number of parasites an entity (Site or host) must contain
        to be included in the analysis
    host_min : int
        The minimum number of hosts necessary for an entity (SITE) to be
        included in the analysis
    output : str
        If True, a string is printed to track progress
    finite : bool
        If True, a finite prediction will be used. This will be Binomial for
        the bottom-up model and cnbinom for the top-down
    model : str
        Either "top-down" (Geommetric/Geom-Gamma/Feasible Set) or "bottom-up"
        (Poisson/Poisson-Gamma).
    heterogeneity : True
        If True, the corresponding top-down or bottom-up model is mixed on
        a gamma-distribution. This results in a negative binomial for the
        bottom-up predictions and a geometric-gamma distribution for the
        top-down. In practice, these are very similar so we always use the
        negative binomial or finite negative binomial.

    Returns
    -------
    : dict
        Each key is a str in split. Looks up a tuple
        Tuple item 1: estimated k parameter
                (will be np.inf if using bottom-up model with no heterogeneity)
        Tuple item 2: predicted vector
        Tuple item 3: observed vector

    """

    # Get unique values in the initial split column.  Often will be site
    sites = para_data[initial_split].unique()

    model_predictions = {}

    for site in sites:

        site_data = para_data[para_data[initial_split] == site]

        if len(site_data) >= host_min:

            if site_data[para_name].sum() >= para_min:

                if output:
                    print("Getting model prediction for {0} for {1}".format(site, para_name))

                # Get the state variables
                P = site_data[para_name].sum()
                H = len(site_data)

                k_range = np.linspace(0.01, 10, num=1000)

                if model == "top-down":

                    if heterogeneity: # Could change to geom_gamma in future
                        if finite:
                            mu, k, b = mod.cnbinom.fit_mle(site_data[para_name],
                                                k_array=k_range)
                        else:
                            mu, k = mod.nbinom.fit_mle(site_data[para_name],
                                            k_array=k_range)
                    else:
                        k = 1

                    pred = nbd_mixture([(P, H)], k=k, finite=finite)

                elif "bottom-up":

                    if heterogeneity:

                        if finite:
                            mu, k, b = mod.cnbinom.fit_mle(site_data[para_name],
                                                k_array=k_range)
                        else:
                            mu, k = mod.nbinom.fit_mle(site_data[para_name],
                                            k_array=k_range)
                    else:
                        k = np.inf

                    pred = poisson_mixture([(P, H)], finite=finite)

                model_predictions[site] = (k, pred,
                            np.sort(np.array(site_data[para_name]))[::-1])

    return model_predictions

################################################

def extract_obs_pred_vectors(vect_dict, model_str, host, parasite):
    """
    Extract the observed and predicted vectors

    Parameters
    ----------
    vect_dict : dict
        The results of manuscript_analysis_no_hetero.py. A dictionary
        with empirical and predicted parasite distributions
    model_str : str
        The model string. "feasible", "trun_geometric", "geometric", "binomial"
        "poisson"
    host : str
        "BUBO", "PSRE", "RACA", "TATO", or "TAGR"
    parasite : str
        "ECSP", "RION", "ALAR", "MANO", "GLOB"

    Returns
    -------
    : tuple
        1. list of observed vectors
        2. Corresponding list of predicted vectors

    """

    all_obs = []
    all_pred = []

    for year in vect_dict.viewkeys():
        for site in vect_dict[year][host][parasite].viewkeys():

            all_obs.append(vect_dict[year][host][parasite][site]['observed'])
            all_pred.append(vect_dict[year][host][parasite][site][model_str])

    return all_obs, all_pred

def extract_obs_pred_stats(stats_dict, model_str, host, parasite, stats):
    """
    Extract statistics

    Parameters
    ----------
    stats_dict : dict
        The results of manuscript_analysis_no_heterogenetity.py. A dictionary
        with summary stats of models
    model_str : str
        The model string. "feasible", "trun_geometric", "geometric", "binomial"
        "poisson"
    host : str
        "BUBO", "PSRE", "RACA", "TATO", or "TAGR"
    parasite : str
        "ECSP", "RION", or "ALAR"
    stats : list
        A list of stats to extract from the analysis

    Returns
    -------
    : tuple
        1. list of observed vectors
        2. Corresponding list of predicted vectors


    """

    all_stats = []

    for year in stats_dict.viewkeys():
        for site in stats_dict[year][host][parasite].viewkeys():

            vals = tuple(np.atleast_1d(stats_dict[year][host][parasite][site][model_str].ix[stats]))
            all_stats.append(vals)

    return all_stats

def extract_sites_with_criteria(stats_dict, stat, criteria, model):
    """
    Extract the sites that meet some criteria (i.e are True)

    Parameters
    -----------
    stats_dict : dict
        A dictionary with the model stats
    stat : str
        A statistic give in the stats df
    criteria : str
        A boolean statement giving the criteria (i.e. '<0.05')
    model : str
        A model column given in the stat dict

    Returns
    -------
    : list
        A list of tuples where each tuple gives the (year, host, parasite, site)
        combo of where the criteria was met

    """

    identity = []

    for year in stats_dict.viewkeys():
        for host in stats_dict[year].viewkeys():
            for parasite in stats_dict[year][host].viewkeys():
                for site in stats_dict[year][host][parasite].viewkeys():
                    stat_df = stats_dict[year][host][parasite][site]

                    if eval("stat_df[model].ix[stat]{0}".format(criteria)):
                        identity.append((year, host, parasite, site))

    return identity

def extract_vectors_given_tuple(val_dict, vect_ids, model):
    """
    Extracts a given model (or column) from various year, host, parasite, site
    combinations given in vect_ids from the dictionary val_dict

    Parameters
    ----------
    val_dict : dict
        An analysis dict of either stats or vectors
    vect_ids : list of tuples
        Each tuple gives a year, host, parasite, site specification
    model : str
        Specifies the column to extract from the year, host, parasite, site
        combination

    Returns
    -------
    : list
        A list of the desired results

    """

    return [val_dict[np.int(year)][host][parasite][site][model] for year, host,
                                                    parasite, site in vect_ids]

def extract_overall_rsq_from_random(random_vects, vect_ids, group_num, model_nm):
    """
    Extract the overall r-squared from different randomized groupings of hosts

    Parameters
    ----------
    random_vects : dict
        A dictionary of randomized vectors from manuscript_analysis_no_heterogenetity
    vect_ids : list of tuples
        Each tuple gives a year, host, parasite, site specification
    group_num : int
        Either 2 or 3
    model_nm : str
        Either 'feasible', 'trun_geometric', or 'binomial'

    Return
    ------
    : list of 2D matrices
    """
    return [random_vects[group_num][site][model_nm] for site in vect_ids]


def extract_var_importance(stats_dict, vect_ids, model):
    """
    Gets an compute the variable importance from stats dict

    Parameters
    ----------
    stats_dict : dict
        An analysis dict of stats
    vect_ids : list of tuples
        Each tuple gives a year, host, parasite, site specification
    model : str
        Specifies the column to extract from the year, host, parasite, site
        combination

    Returns
    -------
    : A DataFrame where the index is the predictor name and importance is the average
        importance of that predictor across vect_ids and se is the standard error

    """

    var_import = [tdf.ix['var_importance'] for tdf in
              extract_vectors_given_tuple(stats_dict, vect_ids, model)]

    # Get average variable importance
    s = pd.DataFrame(np.concatenate(var_import), columns=['predictor', 'importance'])
    s.importance = np.array(s.importance).astype(np.float)
    means = pd.DataFrame(s.groupby("predictor")['importance'].mean().sort_values(ascending=False))

    ses = pd.DataFrame(s.groupby("predictor")['importance'].mean().sort_index() / \
                    s.groupby("predictor")['importance'].size().sort_index())
    ses = ses.rename(columns={0: "se"})

    return(means.join(ses))


def plot_vectors(all_data, all_pred, one_to_one=True, rsqs=None, savename=None,
    proportion=None, ax=None, ci95=None, importance=None, ylim=(-0.5, 3.5),
    xlim=(-0.5, 3.5), gray=False):
    """
    Given observed (all_data) and predicted (all_pred) vectors combine and plot
    in the White et al (2012) style

    Parameters
    ----------
    all_data : array-like
        Vector of observed parasites loads
    all_pred : array-like
        Vector of predicted parasite loads
    one_to_one : bool
        If True, calculates R^2 to the one-to-one line
    rsqs : list or None
        If not None, plots a histogram of site-level rsq values
    proportion : tuple or None
        If not None, the first item is the total number of sites and the second
        item is proportion to which a given model fits.
    ax : None or Matplotlib axis
        If None, creates a new figure and axis and plots it.  Else, plots on
        the provided axis
    ci95 : tuple
        Either None, or contains tuple with 95% CI for overall R2
    importance : DataFrame
        DataFrame with an index being the predictor variables and the columns
        being the mean importance ("importance") and the standard error about
        the mean ("se").
    ylim, xlim : tuples
        Specifies the range of the x an y axises on the plots

    Returns
    -------
    : fig, ax
        Makes a pretty plot

    """

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7), sharex=True, sharey=True)
        given_ax = False
    else:
        given_ax = True

    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    attrib = {"bottom":True, "top":False, "left":False, "right":False,
                "labelbottom":True, "labelleft":False, "labelsize":7}

    ### This code is coloring the points by density

    # Calculate the point density
    x = all_pred
    y = all_data

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    if not gray:
        ax.scatter(np.log10(x + 1), np.log10(y + 1), c=z, s=50,
                        edgecolor='', alpha=0.2)
    else:
        ax.scatter(np.log10(x + 1), np.log10(y + 1), s=50,
                        c=z, alpha=0.8, edgecolor="black", cmap="Greys")

    vals = np.linspace(0, np.max((x, y)), 100)
    ax.plot(np.log10(vals + 1), np.log10(vals + 1), '--', color="black")

    r_sq = comp.r_squared(y + 1, x + 1, one_to_one=one_to_one, log_trans=True)

    if ci95:
        ax.text(0.5, 0.9, r"Overall $R^2$ = {0}".format(round(r_sq, 2)) +
            "\n95% interval: ({0}, {1})".format(*np.round(ci95, 2)),
            size=15, horizontalalignment="center", transform=ax.transAxes)

    else:
        ax.text(0.5, 0.9, r"Overall $R^2$ = {0}".format(round(r_sq, 2)),
            size=15, horizontalalignment="center", transform=ax.transAxes)


    ax.set_xlabel("ln(Predicted + 1)")

    ax.set_ylabel("ln(Observed + 1)")

    if ylim != None:
        ax.set_ylim(ylim)
    if xlim != None:
        ax.set_xlim(xlim)

    if rsqs is not None:

        sub_ax = fig.add_axes([0.7, 0.2, 0.25, 0.2])
        sub_ax.tick_params(**attrib)

        sub_ax.hist(rsqs, color="grey", bins=np.linspace(-1, 1, num=10))
        sub_ax.set_title(r"$R^2$ for each distribution", size=9)
        sub_ax.set_xlim(-1, 1)

    if proportion is not None:

        ax.text(0.5, 0.02,
            "# of Sites = {0}, Proportion not rejected = {1:.2f}".format(*proportion),
            ha="center",
            transform=ax.transAxes)

    if importance is not None:

        sub_ax = fig.add_axes([0.2, 0.71, 0.25, 0.15])
        importance['importance'].plot(kind="bar", color="grey", yerr=importance.se)
        sub_ax.tick_params(right=None, top=None, labelsize=7)
        sub_ax.set_xlabel("")
        sub_ax.set_ylabel("Average predictor importance", size=8)
        sub_ax.set_ylim(0, 0.6)

    if not given_ax:
        plt.tight_layout()

        if savename:
            fig.savefig(savename)

        return fig, ax
    else:
        return ax


def get_null_mixture(data, host_groups, num_samples=200):
    """
    A function to determine whether the mixture obtained by the regression
    tree approach is a "better" fit than one we would obtain by just randomly
    splitting the data into some number of groups

    Parameters
    ----------
    data : array-like
        Host parasite data
    host_groups : list
        List of number of hosts in each group. The len(host_groups) specifies
        the number of groups and the number at each index specifies the number
        of hosts in each group.
    num_samples : int
        Number of resamples of the random vector

    Returns
    --------
    : combos
        A list of host-parasite tuples of the form [(P, H), (P, H)] for the
        given grouping.

    """

    assert len(data) == sum(host_groups), \
                "Length of data and sum of host groups should match"

    H = np.sum(host_groups)

    combos = []

    for i in xrange(num_samples):

        # Reshuffle data
        shuf_data = np.random.choice(data, len(data), replace=False)

        # Split data
        splits = np.split(shuf_data, np.cumsum(host_groups)[:-1])

        Ps = [np.sum(s) for s in splits]

        combos.append(zip(Ps, host_groups))

    return combos


def feasible_k(P, H, num=100, zeros=True):
    """
    Calculates the mean k for a feasible set given some number of random draws
    from the feasible set.  Returns both the MLE prediction of k and the moment
    estimate.  The moment estimate is based on the sample-size corrected
    esimate from Elliot 1977 and Gregory and Woolhouse 1993.

    Parameters
    ----------
    P : int
        number of parasites
    H : int
        Number of hosts
    num : int
        Number of feasible samples

    Returns
    -------
    : tuple
        First object is mle params, second object is correct moment params


    """

    parts = pyp.pypartitions.rand_partitions(P, H, num, zeros=zeros)

    moment = lambda x: (np.mean(x)**2 - (np.var(x, ddof=1) / len(x))) / \
                            (np.var(x, ddof=1) - np.mean(x))

    mle_param = np.array([mod.nbinom.fit_mle(tp, k_array=np.linspace(0.01, 10, 100))[1] for tp in parts])
    moment_param = np.array([moment(tp) for tp in parts])


    return mle_param, moment_param

## STATS FUNCTIONS ##

def BIC(data, model, params=None, corrected=True):
    """
    Bayesian Information Criteria given data and a model

    Parameters
    ----------
    data : array-like
        Data
    model : frozen scipy dist object
    params : int
        Number of parameters in the model. If None, calculated from model
        object.

    """
    n = len(data)  # Number of observations
    L = comp.nll(data, model)

    if not params:
        k = len(model.kwds) + len(model.args)
    else:
        k = params

    bic_value = np.log(n)*k + 2 * L

    return bic_value

def anderson_darling(obs, pred, nsim=10000):
    """
    Bootstrapped Anderson-Darling test for discrete data.
    Compares observed and predicted distributions. Is a bit more sensitive
    than the KS test, though is still non-parameteric.

    Using ad.test in kSamples package in R

    Parameters
    ----------
    obs : array-like
        Observed distribution
    pred : array-like
        Predicted distribution
    nsim : int
        Number of simulation for AD test

    Returns
    -------
    : float
        p_value testing the null hypothesis that the two distributions are the
        same

    """

    obs = np.array(obs)
    pred = np.array(pred)
    robjects.globalenv['obs'] = robjects.FloatVector(obs)
    robjects.globalenv['pred'] = robjects.FloatVector(pred)
    robjects.globalenv['nsim'] = nsim

    robjects.r("""
        res = ad.test(obs, pred, method='simulated', Nsim=nsim)
        p_val = res$ad[2, 4] # Version 2...not sure why?")
        """)

    return np.array(robjects.globalenv['p_val'])[0]

def adj_r_sq(obs, pred, groups):
    """Calculates adjusted R^2"""

    n = len(obs)

    return 1 - ((comp.sum_of_squares(obs, pred) / (n - groups)) /
                (comp.sum_of_squares(obs, np.mean(obs)) / (n - 1)))


## Constrained feasible sets ##

def constrained_ordered_set(P, H, weight_fxn, proposal_fxn, proposal_ratio,
            samples=10, max_iter=10000, sampling="metropolis-hastings", **kwargs):
    """
    Draws a feasible set given some weighting function on the configuration

    By default uses the metropolis algorithm. Make sure to discard the first
    samples. Can also use the rejection algorithm. But rejection rates tend
    to be very high for larger dimension vectors, making this not the best
    choice.

    Parameters
    ----------
    P : int
        Number of parasites
    H : int
        Number of hosts
    weight_fxn : fxn
        A weight function that takes in a vector and return some probability.
        For example, use the maxent_weight_fxn. The probability can be up to
        some constant.
    proposal_fxn : fxn
        Proposal distribution from which to propose samples
    proposal_ratio : function
        Calculates the ratio of probabilities of draws from the proposal fxn.
        Needed for non-symmetrical proposal fxns.
    samples : int
        The desired number of samples to draw from the weighted feasible set
    max_iter : int
        Maximum number of iterations before breaking form the while loop
    sampling : str
        If "rejection" uses rejection sampling (can be slow).
        If "metropolis-hastings"
        uses metropolis-hasthings algorithm which is much faster.

    Returns
    -------
    : 2D np.array, rejection
        The sample/chain from the given algorithm, number of rejections
    """

    if sampling == "rejection": # Accept/reject algorithm

        fs_samples, rejection = _rejection_algorithm(P, H, weight_fxn,
                                        proposal_fxn, samples, max_iter)

    elif sampling == "metropolis-hastings": # Metropolis-Hastings algorithm

        fs_samples, rejection = _metropolis_hastings(P, H, weight_fxn,
                                proposal_fxn, proposal_ratio, samples)

    else:
        raise NotImplementedError("{0} algorithm is not implemented".format(sampling))

    return np.array(fs_samples), rejection


def _rejection_algorithm(P, H, weight_fxn, proposal_fxn, samples, max_iter):
    """
    Implement rejection algorithm

    Take in (P)arasites and (H)osts and return a sampled distribution
    based on the given weights.  The weight_fxn in this case must return
    a probability!

    Parameters
    ----------
    P : int
        Total number of parasites
    H : int
        Total number of hosts
    weight_fxn : function
        Function specifying the unnormalized probability of observing a vector
        sorted vector.
    proposal_fxn : function
        Proposal distribution from which to draw

    Returns
    -------
    : samples, rejection rate
        Rejection algorithm samples and number of rejections

    """

    fs_samples = []
    rejection = 0
    count = 0

    while len(fs_samples) < samples:

        if count < max_iter:

            tfs = proposal_fxn(P, H)
            weight = weight_fxn(tfs)

            if weight > np.random.random():
                fs_samples.append(tfs)
            else:
                rejection += 1

        else:
            print("Maximum number of iterations exceeded")
            break

        count += 1

    return fs_samples, rejection

def _metropolis_hastings(P, H, weight_fxn, proposal_fxn, proposal_ratio,
                                                        samples):
    """
    Metropolis-Hastings algorithm for sampling (P)arasite/(H)ost vectors

    Parameters
    ----------
    P : int
        Total number of parasites
    H : int
        Total number of hosts
    weight_fxn : function
        Function specifying the unnormalized probability of observing a vector
        sorted vector.
    proposal_fxn : function
        Proposal distribution from which to draw
    proposal_ratio : function
        Calculates the ratio of probabilities of draws from the proposal fxn.
        Needed for non-symmetrical proposal fxns.
    samples : int
        Number of samples to draw

    Returns
    -------
    : samples, rejection rate
        Metropolis-Hastings samples and number of rejections
    """

    fs_samples = []
    rejection = 0

    current = proposal_fxn(P, H)
    current_prob = weight_fxn(current)
    fs_samples.append(current)

    for i in np.arange(samples):

        propose = proposal_fxn(P, H)
        propose_prob = weight_fxn(propose)

        # acceptance ratio based on Metropolis Hastings
        ratio = (propose_prob / current_prob) * proposal_ratio(current, propose)

        if ratio > 1:
            fs_samples.append(propose)
        else:
            rand = np.random.random()

            if ratio > rand:
                fs_samples.append(propose)
            else:
                fs_samples.append(current)
                rejection += 1

        current = fs_samples[i + 1]

        current_prob = weight_fxn(current)

    return fs_samples, rejection


def feasible_proposal(P, H):
    """ Use feasible set as a proposal distribution """

    return(pyp.rand_partitions(P, H, 1, zeros=True)[0])


def feasible_ratio(previous, proposed):
    """ Probability ratio between feasible samples.  Always 1"""

    return(1)


def binomial_proposal(P, H):
    """ Use the binomial/multinomial as a proposal distribution. Assumes equal
    weight on all the hosts/cells
    """

    pred = np.random.multinomial(P, np.repeat(1 / H, H), 1).flatten()
    return(np.sort(pred)[::-1])


def binomial_ratio(previous, proposed):
    """ Probability ratio between binomial samples """

    return(binomial_weight_fxn(previous) / binomial_weight_fxn(proposed))


def maxent_weight_fxn(vect):
    """
    A weighting function that calculates the probability of observing a given
    configuration under the maximum entropy model with a constraint on the mean

    Parameters
    ----------
    vect : array
        A vector (e.g. rank abundance distribution) on which to calculate the
        the maxent probability

    Returns
    -------
    : float
        Probability

    Notes
    -----
    This probability comes from Appendix 1 of the feasible
    manuscript.  The denominator D is not necessary for the Metropolis-Hastings
    Algorithm

    """
    P = np.sum(vect)
    H = len(vect)

    # Calculate the denominator
    D = np.exp(gammaln(H + P) - (gammaln(P + 1) + gammaln(H)))

    hs = np.array(pd.Series(vect).value_counts())
    b = np.exp(gammaln(H + 1) - np.sum(gammaln(hs + 1)))

    return b / D

def maxent_count(vect):
    """Just computes b from equation above"""
    P = np.sum(vect)
    H = len(vect)

    # Calculate the denominator
    hs = np.array(pd.Series(vect).value_counts())
    b = np.exp(gammaln(H + 1) - np.sum(gammaln(hs + 1)))

    return b

def binomial_weight_fxn(vect):
    """
    The probability of a given vector under a binomial distribution. This the
    probability without the denominator because we don't need it

    Parameters
    ----------
    vect : array
        A vector (e.g. rank abundance distribution) on which to calculate the
        the binomial probability

    """

    P = np.sum(vect)
    num = gammaln(P + 1)
    denom = np.sum([gammaln(v + 1) for v in vect])
    return(np.exp(num - (denom)))


def pihm_weight(vect, a, b):
    """
    The probability of a given vector assuming parasite-induced host mortality
    (PIHM)

    The function assumes a logistic probability fxn for the host survival
    function determined by parameters a (threshold) and
    b (degree of parasite pathogenicity). Assumes hosts are independent which
    is a relatively common assumption.

    Parameters
    -----------
    vect : array-like
        Host-parasite configuration/distribution
    a : float
        Threshold parameter in logistic growth function (intercept)
    b : float
        pathogenicity parameter in logistic growth function (slope)

    Returns
    -------
    : float
        A weight for the vector
    """

    vect = np.array(vect)
    vect_prob = np.prod(np.exp(a + b * vect) / (1 + np.exp(a + b * vect)))
    return vect_prob


def pihm_binom_weight(vect, a, b):
    """ Combination of binomial weight and PIHM weight """

    return(pihm_weight(vect, a, b) * binomial_weight_fxn(vect))

def pihm_maxent_weight(vect, a, b):
    """ Combination of binomial weight and PIHM weight """

    return(pihm_weight(vect, a, b ) * maxent_weight_fxn(vect))

## OTHER WEIGHTING FUNCTIONS CAN BE DEFINED SIMILAR TO THE WEIGHT FUNCTIONS ABOVE

## HELPER FUNCTIONS ##

def convert_para_host_to_mixed(para_host_vect):
    """
    Convert a parasite host vector into mixture model parameters

    Parameters
    -----------
    para_host_vect : list of tuples
        parasite, host tuples. [(P1, H1), (P2, H2)...]

    Returns
    --------
    : tuple
        1. mus : means of the mixture models
        2. pis : mixing proportions of the mixture models
    """
    paras, hosts = zip(*para_host_vect)

    p = np.array(paras)
    h = np.array(hosts)

    mus = np.array(p) / np.array(h)
    pis = np.array(h) / float(np.sum(h))

    return mus, pis

def convert_em_results(mus_pis, H):
    """
    Converts EM results to a host-parasite vector

    Parameters
    ----------
    mus_pis : tuple
        First item is a array-like objects with means.
        Second item is an array-like objects is mixing proportions
    H : int
        Total number of hosts

    Returns
    -------
    : list or tuples
        A para_host vect
    """

    hosts = mus_pis[1] * H
    parasites = mus_pis[0] * hosts

    # Correct any rounding problems with geom hosts
    hosts = fix_rounding_errors(hosts)
    parasites = fix_rounding_errors(parasites)


    return convert_to_int(zip(parasites, hosts))


def fix_rounding_errors(vector):
    """
    Helper function to make sure vector sums to total

    Adds or subtracts the necessary amount from the maximum entry

    """

    total = np.sum(vector)
    vector = np.round(vector, decimals=0)
    diff = np.sum(vector) - total

    max_ind = np.argmax(vector)
    if diff < 0:

        vector[max_ind] = vector[max_ind] + np.abs(diff)

    elif diff > 0:

        vector[max_ind] = vector[max_ind] - np.abs(diff)

    else:

        pass

    return vector

def convert_to_int(para_host_vec):
    """
    Takes in a list of tuples and makes sure every item is a integer
    """

    para, host = zip(*para_host_vec)
    H = np.sum(host)
    para_round = np.round(para, decimals=0)
    host_round = np.round(host, decimals=0)

    # Make sure the vector still adds up. Should only miss by one (need to test)
    if np.sum(host) < H:
        ind = np.argmax(np.array(host) - np.floor(host))
        hosts_round[ind] = host_round[ind] + 1

    elif np.sum(host) > H:
        ind = np.argmin(np.array(host) - np.floor(host))
        hosts_round[ind] = host_round[ind] - 1


    return(zip(para_round.astype(np.int), host_round.astype(np.int)))

def dict_merge(old_results, new_results):
    """
    Merge two dictionaries to create a new dict. These dictionaries are from
    the various manuscript analyses with form year, host, parasite, site

    """

    for year in old_results.viewkeys():
        for host in old_results[year].viewkeys():
            for parasite in new_results[year][host].viewkeys():

                old_results[year][host][parasite] = \
                                            new_results[year][host][parasite]

    return old_results



### EM ALGORITHMS ###

def pois(x, mu):
    """ Pois pmf in exponential form """

    x = np.atleast_1d(x)
    vals = np.exp(x * np.log(mu) - mu - gammaln(x + 1))

    # A bit of a hack to ensure that the Poisson MLE will actually work
    #vals[vals == 0] = 1e-50

    return vals

def pois_mixed(x, pis, mus):
    """
    Mixed pois

    Parameters
    ----------
    x : single x value

    pis : array-like
        Prior probability of g groups

    mus : array-like
        Means of g groups

    """

    pmf = np.sum([tp * pois(x, tmu) for tp, tmu in zip(pis, mus)])
    return pmf

def geom(x, mu):
    """ Geometric pmf in exponential form """

    return np.exp(x * np.log(mu / (1 + mu)) - np.log(1 + mu))

def geom_mixed(x, pis, mus):
    """
    Mixed Geometric

    Parameters
    ----------
    x : single x value

    pis : array-like
        Prior probability of g groups

    mus : array-like
        Means of g groups

    """

    pmf = np.sum([tp * geom(x, tmu) for tp, tmu in zip(pis, mus)])
    return pmf


def geom_mixed_pmf(x, pis, mus):
    """ Vectorized version of geom_mixed """
    x = np.atleast_1d(x)
    pmf = np.array([geom_mixed(tx, pis, mus) for tx in x])
    return pmf

def geom_mixed_cdf(x, pis, mus):
    """ """
    x = np.atleast_1d(x)

    max_x = np.max(x)
    pmf_list = geom_mixed_pmf(np.arange(1, np.int(max_x) + 1), pis, mus)

    full_cdf = np.cumsum(pmf_list)

    cdf = np.array([full_cdf[tx - 1] if tx != 0 else 0 for tx in x])
    return cdf

def full_log_likelihood(data, pis, mus, dist):
    """
    Full likelihood for a mixture models.

    Parameters
    ----------
    data : array-like
    pis : array-like
        Prior probability of g groups
    mus : array-like
        Means of g groups
    dist : name
        "geom", "pois", "feasible"

    Returns
    -------
    : float
        The log-likelihood of the mixture model


    """
    data = np.atleast_1d(data)

    if dist == "geom":

        ll = np.sum([np.log(geom_mixed(td, pis, mus)) for td in data])

    elif dist == "pois":

        ll = np.sum([np.log(pois_mixed(td, pis, mus)) for td in data])
    else:
        ll = np.nan

    return ll


def bic(loglike, params, n):
    """
    BIC criteria
    """
    return -2 * loglike + params * np.log(n)

def aic(loglike, k, n, corrected=True):
    """
    AIC criteria

    Parameters
    -----------
    loglike : float
        log-likelihood
    k : float
        Number of estimated parameters
    n : float
        Sample size
    corrected : bool
        If True used AICc, otherwise uses AIC.
    """

    if corrected:
        return 2 * k + -2 * loglike + (2 * k * (k + 1)) / (n - k - 1)
    else:
        return -2 * loglike + k * 2


