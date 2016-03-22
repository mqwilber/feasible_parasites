import agg_fxns as agg
reload(agg)
import numpy as np
import pandas as pd
import macroeco.compare as comp
import scipy.stats as stats
import macroeco.models as mod
import multiprocessing as mp
import logging
logging.basicConfig(filename="continuous_hetero.log", level=logging.DEBUG,
                                        format='%(asctime)s %(message)s')

logger = logging.getLogger("my logger")


"""
Description
-----------

Analyzing Pieter Johnson's macroparasite data across multiple years,
host species, and parasites. Fitting a model with continuous heterogeneity
(i.e. a Negative Binomial model and Finite Negative Binomial model)
to each distribution. This is standard practice when analyzing host-parasite
distributions.  These distributions can then be analyzed like any of the
distributions with continuous heterogeneity computed in
manuscript_analysis_discrete_known_heterogeneity.

Data description
----------------

td_bu_predictions : a dict with the following nested key levels
    1. year (the year of interest)
    2. host (the host species begin looked at)
    3. parasite (the parasite species being looked at)
    4. site (the site being looked at)

    td_bu_predictions[year][host][parasite][site] looks up a dataframe
    with

"""


def get_top_down_bottom_up_model_predictions(processes, para_data, years,
                                hosts, parasites, splitby, para_min, host_min):
    """
    Multiprocessing

    """

    pool = mp.Pool(processes=processes)

    results = [pool.apply_async(_get_td_bu_predictions_by_year,
                args=(para_data[para_data.year == year], hosts, parasites,
                      splitby, para_min, host_min)) for year in years]

    results = [p.get() for p in results]
    results.sort() # to sort the results
    return dict(results)

def _get_td_bu_predictions_by_year(year_data, hosts,
                                    parasites, splitby, para_min, host_min):
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

    Returns
    ------
    : dict


    """

    td_bu_predictions = {}
    year = year_data.year.unique()[0]

    for host in hosts:

        logging.info("Year {0}: Beginning host {1}".format(year, host))
        td_bu_predictions[host] = {}

        sub_data = year_data[year_data.speciescode == host]

        for parasite in parasites:

            logging.info("Year {0}: Beginning parasite {1}".format(year, parasite))

            # Get finite top-down model predictions
            top_down_pred_finite = \
                    agg.get_model_predictions_from_data(sub_data,
                        parasite, splitby, para_min=para_min,
                        host_min=host_min, model="top-down", output=False,
                        finite=True, heterogeneity=True)

            # Get infinite top-down model predictions
            top_down_pred_infinite = \
                    agg.get_model_predictions_from_data(sub_data,
                        parasite, splitby, para_min=para_min,
                        host_min=host_min, model="top-down", output=False,
                        finite=False, heterogeneity=True)


            td_bu_predictions[host][parasite] = {}

            # Save all results for each site in a pandas dataframe
            for site in top_down_pred_finite.viewkeys():

                td_bu_predictions[host][parasite][site] = \
                    pd.DataFrame(zip(
                    top_down_pred_finite[site][2],
                    top_down_pred_finite[site][1],
                    top_down_pred_infinite[site][1],
                    np.repeat(top_down_pred_finite[site][0], len(top_down_pred_finite[site][1])),
                    np.repeat(top_down_pred_infinite[site][0], len(top_down_pred_finite[site][1]))),
                    columns = ['observed',
                               'finite_nbd',
                               'nbd',
                               'finite_nbd_k',
                               'nbd_k'])


    logging.info("Completed year {0}".format(year))
    return (year, td_bu_predictions)


def get_top_down_bottom_up_model_stats(pred_dict):
    """
    Takes in the prediction dictionary from
    get_top_down_bottom_up_model_predictions and calculates various statistics
    for each of the predicted vectors

    Parameters
    ----------
    pred_dict : dict
        Results from get_top_down_bottom_up_model_predictions

    """

    td_bu_summary_stats = {}

    years = pred_dict.viewkeys()

    for year in years:

        logging.info("Year {0}: Getting stats".format(year))

        td_bu_summary_stats[year] = {}
        hosts = pred_dict[year].viewkeys()

        for host in hosts:

            td_bu_summary_stats[year][host] = {}
            parasites = pred_dict[year][host].viewkeys()

            for parasite in parasites:

                td_bu_summary_stats[year][host][parasite] = {}
                sites = pred_dict[year][host][parasite].viewkeys()

                for site in sites:

                    # Extract the prediction DataFrame
                    pred_df = pred_dict[year][host][parasite][site]
                    obs = pred_df['observed']
                    pred_columns = pred_df.columns[1:3]
                    k_finite = pred_df['finite_nbd_k'].iloc[0]
                    k_infinite = pred_df['nbd_k'].iloc[0]

                    # Empty DataFrame to save results
                    stats_df = pd.DataFrame(
                                    index=['r_sq_one_to_one',
                                           'AIC',
                                           'BIC',
                                           'AD_p_val'],
                                    columns=['finite_nbd',
                                             'nbd'])

                    r_sqs_1to1s = [comp.r_squared(obs + 1, pred_df[col] + 1,
                                    one_to_one=True, log_trans=True) for col in
                                    pred_columns]
                    stats_df.ix['r_sq_one_to_one'] = r_sqs_1to1s

                    # Anderson Darling Test
                    ad_p_vals = [agg.anderson_darling(obs, pred_df[col]) for
                                        col in pred_columns]
                    stats_df.ix['AD_p_val'] = ad_p_vals

                    # Calculate AIC, BIC for each model but feasible set
                    models = ["mod.cnbinom(mu={0}, k_agg={1}, b={2})",
                              "mod.nbinom(mu={0}, k_agg={1})"]
                    P = np.sum(obs)
                    H = np.float(len(obs))

                    params = [((P / H), k_finite, P), (P / H, k_infinite)]


                    AICS = [comp.AIC(obs, eval(m.format(*p)), params=len(p))
                                for m, p in zip(models, params)]

                    BICS = [agg.BIC(obs, eval(m.format(*p)), params=len(p))
                                for m, p in zip(models, params)]

                    stats_df.ix['AIC'] = AICS
                    stats_df.ix['BIC'] = BICS

                    # TODO: Check the correspondence between r_sq and AIC
                    # They are measuring different types of fit!
                    # Overall R^2 and proportion of sites that are good fits
                    # correspond very well


                    td_bu_summary_stats[year][host][parasite][site] = stats_df
        logging.info("Completed Year {0}".format(year))

    return td_bu_summary_stats


if __name__ == '__main__':

    # Load in the data
    para_data = pd.read_csv("../data/archival/dummy_data.csv")

    ## PARAMETERS FOR ANALYSIS ##

    years = [2009, 2010, 2011, 2012, 2013, 2014] # years to include
    hosts = ["BUBO", "PSRE", "RACA", "TATO", "TAGR"] # hosts to include
    parasites = ["GLOB", "MANO", "RION","ECSP", "ALAR"] # parasites to include
    splitby = "sitename" # SITE string

    para_min = 10 # Minimum number of parasites
    host_min = 10 # Minimum number of hosts

    #########################################

    # 1. Loop through the datasets to get the fitted models for top-down and
    # bottom-up predictions with no heterogeneity. Use multiprocessing to speed
    # things up


    td_bu_predictions = get_top_down_bottom_up_model_predictions(3, para_data,
                            years, hosts, parasites, splitby, para_min,
                            host_min)

    td_bu_summary_stats = get_top_down_bottom_up_model_stats(td_bu_predictions)


    # Save the vectors
    pd.to_pickle(td_bu_predictions,
               "../results/pickled_results/continuous_heterogeneity_all_years_vectors.pkl")

    # 2. Calculate a variety of statistics for each site: AIC, BIC, KS-test p-value,
    # r_squared to the one_to_one line.

    pd.to_pickle(td_bu_summary_stats, "../results/pickled_results/continuous_heterogeneity_all_years_stats.pkl")
