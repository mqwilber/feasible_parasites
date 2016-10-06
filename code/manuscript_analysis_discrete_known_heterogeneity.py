import agg_fxns as agg
reload(agg)
import numpy as np
import pandas as pd
import macroeco.compare as comp
import scipy.stats as stats
import macroeco.models as mod
import multiprocessing as mp
# import logging
# logging.basicConfig(filename="discrete_known_hetero.log", level=logging.DEBUG,
#                                         format='%(asctime)s %(message)s')

# logger = logging.getLogger("my logger")
SAMPLES = 1000 # Samples for feasible set

"""
Description
-----------

Analyzing Pieter Johnson's macroparasite data across multiple years,
host species, and parasites.

This script is accounting for known host-level covariates and exploring how
these covariates might affect the distribution of parasites across hosts. The
covariates that are being considered are host body size and the loads of
other trematode parasites.  This script considers 5 different analyses

    1. Predictor variables are host body size and individual trematode loads
    2. Predictor variable is just host body size
    3. Predictor variables are only individual trematode loads
    4. Predictor variables are host body size and total trematode load
    5. Predictor variable is only total trematode load

Data description (1 for each of the 5 analyses)
-----------------------------------------------

td_bu_predictions : a dict with the following nested key levels
    1. year (the year of interest)
    2. host (the host species begin looked at)
    3. parasite (the parasite species being looked at)
    4. site (the site being looked at)

    td_bu_predictions[year][host][parasite][site] looks up a dataframe
    with columns 'observed', 'feasible', 'trun_geometric', 'geometric',
    'binomial', and 'poisson', where each column contains the predicted rank
    abundance distribution for the specified model.

 td_bu_summary_stats : A dict of similar structure as above

    td_bu_summary_stats[year][host][parasite][site] looks up a dataframe with
    columns 'feasible', 'trun_geometric', 'geometric', 'binomial', and
    'poisson' where each column contains various summary statistics on how
    well the predicted distribution fits the data.

"""


def get_top_down_bottom_up_model_predictions(processes, para_data, years,
                                hosts, parasites, splitby, para_min, host_min,
                                pred_type):
    """
    Multiprocessing function to speed things up.
    """

    pool = mp.Pool(processes=processes)

    results = [pool.apply_async(_get_td_bu_predictions_by_year,
                args=(para_data[para_data.year == year], hosts, parasites,
                      splitby, para_min, host_min, pred_type)) for year in years]

    results = [p.get() for p in results]
    results.sort() # to sort the results

    # Clean up processes
    pool.close()
    pool.join()

    years, vectors, stats = zip(*results)

    return dict(zip(years, vectors)), dict(zip(years, stats))

def _get_td_bu_predictions_by_year(year_data, hosts,
                        parasites, splitby, para_min, host_min, pred_type,
                        num_groups=5):
    """
    Function constructed for multiprocessing. The multiprocessing is happening
    on the year level to speed the analysis up.

    Parameters
    ----------
    year_data : DataFrame
        Parasite data subsetted on a given year
    hosts : list
        List of hosts
    parasites : list
        List of parasites
    splitby : str
        String on which to split the data
    para_min : float
        Minimum parasite load at a give split necessary to include sample
    host_min : float
        Minimum hosts at a given split necessary to include sample
    pred_type : str
        Type of predictor variables to use in the cart analysis
    num_groups : int
        Number of heterogeneity groups to use

    Returns
    ------
    : dict


    """

    td_bu_predictions = {}
    td_bu_stats = {}
    year = year_data.year.unique()[0]

    for host in hosts:

        logging.info("Year {0}: Beginning host {1}".format(year, host))
        td_bu_predictions[host] = {}
        td_bu_stats[host] = {}

        sub_data = year_data[year_data.speciescode == host]

        for parasite in parasites:

            # Set predictor variables:
            up_sub_data, predictors = get_predictors(parasites, parasite,
                                                        sub_data, pred_type)

            # parasite_pred = list(np.copy(parasites))
            # parasite_pred.remove(parasite)
            # predictors = np.sort(["svl"] + parasite_pred)

            sites = up_sub_data[splitby].unique()

            td_bu_predictions[host][parasite] = {}
            td_bu_stats[host][parasite] = {}

            for site in sites:

                site_data = up_sub_data[up_sub_data[splitby] == site]

                if (len(site_data) >= host_min) and \
                            (site_data[parasite].sum() >= para_min):

                    logging.info("Year {0}: Beginning parasite {1}, site {2}".format(year, parasite, site))
                    obs = np.sort(site_data[parasite])[::-1]

                    groups = np.arange(1, num_groups + 1)

                    # Empty dataframe
                    cols = ["observed"] + \
                            ["feasible_{0}".format(g) for g in groups] +\
                            ["trun_geometric_{0}".format(g) for g in groups] +\
                            ["geometric_{0}".format(g) for g in groups] +\
                            ["binomial_{0}".format(g) for g in groups] +\
                            ["poisson_{0}".format(g) for g in groups]
                    vect_res = pd.DataFrame(index=np.arange(0, len(obs)),
                        columns=cols)
                    stats = ['r_sq_one_to_one', 'AIC', 'BIC', 'AD_p_val',
                                                            'var_importance']
                    stats_res = pd.DataFrame(index=stats, columns=cols[1:])

                    # Save and sort observed vector
                    vect_res['observed'] = np.sort(obs)[::-1]

                    # Loop through each group
                    for group in groups:

                        dists = {'feas' : ('feasible', agg.feasible_mixture),
                                 'geom' : ('geometric', 'trun_geometric',
                                                            agg.nbd_mixture),
                                 'pois' : ('poisson', 'binomial',
                                                        agg.poisson_mixture)}

                        for dist in dists.viewkeys():

                            # Get initial predictions
                            if group != 1:
                                ph_pred, tree = agg.cart_analysis(site_data,
                                    parasite, predictors, max_leaf_nodes=group)

                                importance = tree.feature_importances_
                                df_import = zip(predictors, importance)

                            else:
                                ph_pred = [(np.sum(obs), len(obs))]
                                df_import = [(np.nan, np.nan)]

                            mus, pis = agg.convert_para_host_to_mixed(ph_pred)

                            ll = agg.full_log_likelihood(obs, mus, pis, dist)

                            if dist != "feas":

                                # Infinite predictions
                                pred_infinite = dists[dist][2](ph_pred,
                                                                  finite=False)
                                infinite_stats = \
                                    get_stats(obs, pred_infinite, ll, group*2,
                                                                    df_import)

                                infinite_col = "{0}_{1}".format(dists[dist][0],
                                                                        group)
                                vect_res[infinite_col] = pred_infinite
                                stats_res[infinite_col] = infinite_stats

                                # Finite predictions
                                pred_finite = dists[dist][2](ph_pred,
                                                                   finite=True)
                                finite_stats = get_stats(obs, pred_finite, ll,
                                        group*2 + 1, df_import)

                                finite_col = "{0}_{1}".format(dists[dist][1],
                                                                         group)
                                vect_res[finite_col] = pred_finite
                                stats_res[finite_col] = finite_stats

                            else:

                                # Group one is a very intensive feasible set
                                # calculation and has already been calculated
                                # elsewhere...don't do it again
                                if group != 1:

                                    # Finite predictions for feasible set
                                    pred_finite = dists[dist][1](ph_pred,
                                               samples=SAMPLES,
                                               center="median")[1]
                                    finite_stats = get_stats(obs, pred_finite,
                                            ll, group*2 + 1, df_import)

                                    finite_col = "{0}_{1}".format(
                                                         dists[dist][0], group)
                                    vect_res[finite_col] = pred_finite
                                    stats_res[finite_col] = finite_stats

                    td_bu_predictions[host][parasite][site] = vect_res
                    td_bu_stats[host][parasite][site] =  stats_res

    logging.info("Completed year {0}".format(year))
    return (year, td_bu_predictions, td_bu_stats)

def get_predictors(parasites, parasite, sub_data, pred_type):
    """
    Returns a list of predictors based on which predictors to use.  If
    necessary, creates a new column in the data

    Parameters
    ----------
    parasites : list
        List of parasite names
    parasite : str
        Focal parasite
    sub_data : DataFrame
        The year specific parasite data
    pred_type : str
        "all_para": predictors are "svl" + unique parasites
        "only_svl": predictors are only "svl"
        "only_para": predictors are only unique parasites
        "all_total": predictors are "svl" and total number of trematodes
        "only_total": predictors are only total number of parasites

    Returns
    -------
    : sub_data, predictors
        An updated sub_data dataframe and a list of predictors
    """

    # Functions to specify predictors
    def all_para_fxn(parasites, parasite, sub_data):
        parasite_pred = list(np.copy(parasites))
        parasite_pred.remove(parasite)
        predictors = np.sort(["svl"] + parasite_pred)
        return((sub_data, predictors))

    def only_svl_fxn(parasites, parasite, sub_data):
        predictors = np.array(["svl"])
        return((sub_data, predictors))

    def only_para_fxn(parasites, parasite, sub_data):
        parasite_pred = list(np.copy(parasites))
        parasite_pred.remove(parasite)
        predictors = np.sort(parasite_pred)
        return((sub_data, predictors))

    def all_total_fxn(parasites, parasite, sub_data):
        parasite_pred = list(np.copy(parasites))
        parasite_pred.remove(parasite)
        sub_data['total_para'] = sub_data[parasite_pred].sum(axis=1)
        predictors = np.array(["svl", "total_para"])
        return((sub_data, predictors))

    def only_total_fxn(parasites, parasite, sub_data):
        parasite_pred = list(np.copy(parasites))
        parasite_pred.remove(parasite)
        sub_data['total_para'] = sub_data[parasite_pred].sum(axis=1)
        predictors = np.array(["total_para"])
        return((sub_data, predictors))

    # Switch dictionary
    switch_dict = {'all_para': all_para_fxn,
                   'only_svl': only_svl_fxn,
                   'only_para': only_para_fxn,
                   'all_total': all_total_fxn,
                   'only_total': only_total_fxn}

    pred_fxn = switch_dict.get(pred_type, None)

    if pred_fxn:
        return(pred_fxn(parasites, parasite, sub_data))
    else:
        raise KeyError("Predictor type not recognized")


def get_stats(obs, pred, ll, params, df_import):
    """
    Get 'r_sq_one_to_one', 'AIC', 'BIC', 'AD_p_val'
    """

    rsq = comp.r_squared(obs + 1, pred + 1, one_to_one=True, log_trans=True)
    bic = agg.bic(ll, params, len(obs))
    aic = agg.aic(ll, params, len(obs))
    adpval = agg.anderson_darling(obs, pred)

    return [rsq, aic, bic, adpval, df_import]


def fill_feasible_1(vects_dict, stats_dict):
    """
    Fill feasible_1 columns from already calculated data.

    Parameters
    ----------
    vects_dict : dict
        td_bu_predictions
    stats_dict : dict
        td_bu_summary_stats

    Returns
    -------
    : an updated vects_dict, an updated stats_dict

    """

    # TODO: Check that these files exist!

    # Load in previously calculated data
    no_hetero_vects = pd.read_pickle("../results/pickled_results/no_heterogeneity_all_years_vectors.pkl")
    no_hetero_stats = pd.read_pickle("../results/pickled_results/no_heterogeneity_all_years_stats.pkl")

    for year in vects_dict.viewkeys():
        for host in vects_dict[year].viewkeys():
            for para in vects_dict[year][host].viewkeys():
                for site in vects_dict[year][host][para].viewkeys():

                    vects_dict[year][host][para][site]['feasible_1'] = \
                            no_hetero_vects[year][host][para][site]['feasible']

                    stats_dict[year][host][para][site]['feasible_1'] = \
                            no_hetero_stats[year][host][para][site]['feasible']

    return vects_dict, stats_dict


if __name__ == '__main__':

    import logging
    logging.basicConfig(filename="discrete_known_hetero.log", level=logging.DEBUG,
                                        format='%(asctime)s %(message)s')

    logger = logging.getLogger("my logger")

    # Load in the data
    para_data = pd.read_csv("../data/archival/dummy_data.csv")

    ## PARAMETERS FOR ANALYSIS ##

    years = [2009, 2010, 2011, 2012, 2013, 2014] # years to include
    hosts = ["BUBO", "PSRE", "RACA", "TATO", "TAGR"] # hosts to include
    parasites = ["GLOB", "MANO", "RION", "ECSP", "ALAR"] # parasites to include

    # Different types of heterogeneity analysis. See get_predictors fxn for
    # description of types. These specify the predictors to use in the
    # regression tree analysis
    predictor_types = ["only_svl", "only_para", "all_total", "only_total"]
    #["all_para", "only_svl", "only_para", "all_total", "only_total"]

    splitby = "sitename" # SITE string

    para_min = 10 # Minimum number of parasites
    host_min = 10 # Minimum number of hosts

    #########################################

    # 1. For each predictor type, loop through the datasets to get the fitted
    # models for top-down and bottom-up predictions with no heterogeneity.
    # Use multiprocessing to speed things up

    for pred_type in predictor_types:

        logging.info("Beginning analysis with predictor type {0}".format(pred_type))

        td_bu_predictions, td_bu_summary_stats = \
                get_top_down_bottom_up_model_predictions(3, para_data, years,
                                hosts, parasites, splitby, para_min,
                                host_min, pred_type)

        # This function assumes that manuscript_analysis_no_hetero.py has already
        # been run!
        td_bu_predictions, td_bu_summary_stats = fill_feasible_1(td_bu_predictions,
                                                     td_bu_summary_stats)

        # # Save the vectors
        pd.to_pickle(td_bu_predictions,
            "../results/pickled_results/discrete_known_heterogeneity_all_years_vectors_{0}.pkl".format(pred_type))
        pd.to_pickle(td_bu_summary_stats,
            "../results/pickled_results/discrete_known_heterogeneity_all_years_stats_{0}.pkl".format(pred_type))

        logging.info("Analysis for predictors {0} saved".format(pred_type))

    logging.info("Completed analysis")
