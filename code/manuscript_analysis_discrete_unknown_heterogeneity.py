import agg_fxns as agg
reload(agg)
import numpy as np
import pandas as pd
import macroeco.compare as comp
import scipy.stats as stats
import macroeco.models as mod
import multiprocessing as mp
import logging
logging.basicConfig(filename="discrete_unknown_hetero.log", level=logging.DEBUG,
                                        format='%(asctime)s %(message)s')

logger = logging.getLogger("my logger")


"""
Description
-----------

Analyzing Pieter Johnson's macroparasite data across multiple years,
host species, and parasites. This analysis is focusing on discrete and
unknown levels o heterogeneity in the hosts.

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

    years, vectors, stats = zip(*results)

    return dict(zip(years, vectors)), dict(zip(years, stats))

def _get_td_bu_predictions_by_year(year_data, hosts,
                        parasites, splitby, para_min, host_min, num_groups=5):
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
    td_bu_stats = {}
    year = year_data.year.unique()[0]

    for host in hosts:

        logging.info("Year {0}: Beginning host {1}".format(year, host))
        td_bu_predictions[host] = {}
        td_bu_stats[host] = {}

        sub_data = year_data[year_data.speciescode == host]

        for parasite in parasites:

            sites = sub_data[splitby].unique()

            td_bu_predictions[host][parasite] = {}
            td_bu_stats[host][parasite] = {}

            for site in sites:

                site_data = sub_data[sub_data[splitby] == site]

                if (len(site_data) >= host_min) and \
                            (site_data[parasite].sum() >= para_min):

                    logging.info("Year {0}: Beginning parasite {1}, site {2}".format(year, parasite, site))
                    obs = np.sort(site_data[parasite])[::-1]

                    groups = np.arange(1, num_groups + 1)

                    # Empty dataframe
                    cols = ["observed"] + \
                            ["trun_geometric_{0}".format(g) for g in groups] +\
                            ["geometric_{0}".format(g) for g in groups] +\
                            ["binomial_{0}".format(g) for g in groups] +\
                            ["poisson_{0}".format(g) for g in groups]
                    vect_res = pd.DataFrame(index=np.arange(0, len(obs)),
                        columns=cols)
                    stats = ['r_sq_one_to_one', 'AIC', 'BIC', 'AD_p_val']
                    stats_res = pd.DataFrame(index=stats, columns=cols[1:])

                    # Save and sort observed vector
                    vect_res['observed'] = np.sort(obs)[::-1]

                    # Loop through each group and get
                    for group in groups:

                        dists = {'geom' : ('geometric', 'trun_geometric', agg.nbd_mixture),
                                'pois' : ('poisson', 'binomial', agg.poisson_mixture)}

                        for dist in dists.viewkeys():

                            # Get initial predictions
                            if group != 1:
                                ph_vect, _ = agg.cart_analysis(site_data, parasite,
                                    ["svl"], max_leaf_nodes=group)
                            else:
                                ph_vect = [(np.sum(obs), len(obs))]

                            mus_init, pis_init = agg.convert_para_host_to_mixed(ph_vect)

                            mus, pis, ll, num_iter = \
                                agg.em_algorithm(obs, mus_init, pis_init, dist,
                                                                    tol=0.001)
                            # Only fit if it is not nan
                            if ~np.isnan(ll):

                                ph_pred = agg.convert_em_results((mus, pis), len(obs))

                                # Infinite predictions
                                pred_infinite = dists[dist][2](ph_pred, finite=False)
                                infinite_stats = \
                                    get_stats(obs, pred_infinite, ll, group*2)

                                infinite_col = "{0}_{1}".format(dists[dist][0], group)
                                vect_res[infinite_col] = pred_infinite
                                stats_res[infinite_col] = infinite_stats

                                # Finite predictions
                                pred_finite = dists[dist][2](ph_pred, finite=True)
                                finite_stats = get_stats(obs, pred_finite, ll, group*2 + 1)

                                finite_col = "{0}_{1}".format(dists[dist][1], group)
                                vect_res[finite_col] = pred_finite
                                stats_res[finite_col] = finite_stats


                    td_bu_predictions[host][parasite][site] = vect_res
                    td_bu_stats[host][parasite][site] =  stats_res

    logging.info("Completed year {0}".format(year))
    return (year, td_bu_predictions, td_bu_stats)


def get_stats(obs, pred, ll, params):
    """
    Get 'r_sq_one_to_one', 'AIC', 'BIC', 'AD_p_val'
    """

    rsq = comp.r_squared(obs + 1, pred + 1, one_to_one=True, log_trans=True)
    bic = agg.bic(ll, params, len(obs))
    aic = agg.aic(ll, params, len(obs))
    adpval = agg.anderson_darling(obs, pred)

    return [rsq, aic, bic, adpval]


if __name__ == '__main__':

    # Load in the data
    para_data = pd.read_csv("../data/archival/dummy_data.csv")

    ## PARAMETERS FOR ANALYSIS ##

    years = [2009, 2010, 2011, 2012, 2013, 2014] # years to include
    hosts = ["BUBO", "PSRE", "RACA", "TATO", "TAGR"] # hosts to include
    parasites = ["MANO", "GLOB", "RION", "ECSP", "ALAR"] # parasites to include
    splitby = "sitename" # SITE string

    para_min = 10 # Minimum number of parasites
    host_min = 10 # Minimum number of hosts

    #########################################

    # 1. Loop through the datasets to get the fitted models for top-down and
    # bottom-up predictions with no heterogeneity. Use multiprocessing to speed
    # things up

    #year_data = para_data[para_data.year == 2009]

    td_bu_predictions, td_bu_summary_stats = \
            get_top_down_bottom_up_model_predictions(3, para_data, years,
                            hosts, parasites, splitby, para_min,
                            host_min)

    # Save the vectors
    pd.to_pickle(td_bu_predictions,
        "../results/pickled_results/discrete_unknown_heterogeneity_all_years_vectors.pkl")

    # 2. Calculate a variety of statistics for each site: AIC, BIC, KS-test p-value,
    # r_squared to the one_to_one line.

    pd.to_pickle(td_bu_summary_stats,
        "../results/pickled_results/discrete_unknown_heterogeneity_all_years_stats.pkl")

