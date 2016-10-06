import agg_fxns as agg
reload(agg)
import numpy as np
import pandas as pd
import macroeco.compare as comp
import scipy.stats as stats
import macroeco.models as mod
import multiprocessing as mp
import logging
logging.basicConfig(filename="no_hetero_bs_gof.log", level=logging.DEBUG,
                                        format='%(asctime)s %(message)s')

logger = logging.getLogger("my logger")
SAMPLES = 500

"""
Description
-----------

Analyzing Pieter Johnson's macroparasite data across multiple years,
host species, and parasites. This script is complementary to 
manuscript_analysis_no_hetero.py as it provides an additional measure of GOF
based on a parametric bootstrap of the different models.  See feasible_manuscript.tex
and supplementary_material.tex (Appendix S3) for a description of these tests.

Data description
----------------

td_bu_predictions : a dict with the following nested key levels
    1. year (the year of interest)
    2. host (the host species begin looked at)
    3. parasite (the parasite species being looked at)
    4. site (the site being looked at)

    td_bu_predictions[year][host][parasite][site] looks up a dictionary with
    keywords `feasible`, `trun_geometric`, and `binomial` corresponding to the
    different feasible set models. Each key contains a tuple with the observed
    KS statistics and the distribution of KS statistics based on 500 bootstraps
    from the respective weighted feasible set.


"""


def get_top_down_bottom_up_model_predictions(processes, para_data, years,
                                hosts, parasites, splitby, para_min, host_min):
    """
    A function for multiprocessing to speed up the calculations
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

            # Feasible set predictions

            feasible_pred = agg.get_single_host_predictions_from_data(sub_data,
                parasite, splitby, samples=SAMPLES, para_min=para_min,
                host_min=host_min, output=False, center="median",
                logger=(logger, year), large_samples=SAMPLES,
                fs_type="partition")

            # Maxent/composition predictions
            maxent_pred = agg.get_single_host_predictions_from_data(sub_data,
                parasite, splitby, samples=SAMPLES, para_min=para_min,
                host_min=host_min, output=False, center="median",
                logger=(logger, year), large_samples=SAMPLES,
                fs_type="composition")

            # Multinomial predictions
            multinom_pred = agg.get_single_host_predictions_from_data(sub_data,
                parasite, splitby, samples=SAMPLES, para_min=para_min,
                host_min=host_min, output=False, center="median",
                logger=(logger, year), large_samples=SAMPLES,
                fs_type="multinomial")

            td_bu_predictions[host][parasite] = {}

            # Save all results for each site in a pandas dataframe
            for site in maxent_pred.viewkeys():

                obs_like_fs = np.sum(np.log(feasible_pred[site][1](feasible_pred[site][2])))

                dist_like_fs = np.array([np.sum(np.log(feasible_pred[site][1](x))) for 
                                        x in feasible_pred[site][0]])

                obs_like_me = np.sum(np.log(maxent_pred[site][1](maxent_pred[site][2])))
                
                dist_like_me = np.array([np.sum(np.log(maxent_pred[site][1](x))) for 
                                        x in maxent_pred[site][0]])

                obs_like_mn = np.sum(np.log(multinom_pred[site][1](multinom_pred[site][2])))
                
                dist_like_mn = np.array([np.sum(np.log(multinom_pred[site][1](x))) for 
                                        x in multinom_pred[site][0]])

                td_bu_predictions[host][parasite][site] = \
                    {"feasible": (obs_like_fs, dist_like_fs), 
                     "trun_geometric": (obs_like_me, dist_like_me), 
                      "binomial": (obs_like_mn, dist_like_mn)}


    logging.info("Completed year {0}".format(year))
    return (year, td_bu_predictions)


if __name__ == '__main__':

    # Load in the data
    para_data = pd.read_csv("../data/archival/dummy_data.csv")

    ## PARAMETERS FOR ANALYSIS ##

    years = [2009, 2010, 2011, 2012, 2013, 2014] # years to include
    hosts = ["BUBO", "PSRE", "RACA", "TATO", "TAGR"] # hosts to include
    parasites = ["GLOB", "MANO", "RION", "ECSP", "ALAR"] # parasites to include
    splitby = "sitename" # SITE string

    para_min = 10 # Minimum number of parasites
    host_min = 10 # Minimum number of hosts

    adding = False # True if you are adding new parasites

    # temp = _get_td_bu_predictions_by_year(para_data[para_data.year == 2009], hosts, parasites,
    #                   splitby, para_min, host_min)

    #########################################

    # 1. Loop through the datasets to get the fitted models for top-down and
    # bottom-up predictions with no heterogeneity. Use multiprocessing to speed
    # things up

    td_bu_predictions = get_top_down_bottom_up_model_predictions(3, para_data,
                            years, hosts, parasites, splitby, para_min,
                            host_min)

    pd.to_pickle(td_bu_predictions,
             "../results/pickled_results/no_heterogeneity_all_years_stats_bs_gof.pkl")



