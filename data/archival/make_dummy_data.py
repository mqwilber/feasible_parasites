from __future__ import division
import numpy as np
import pandas as pd
import macroeco.models as md
import scipy.stats as stats
import itertools

"""
Description
-----------

Make dummy data so scripts will run. This will allows users to play around
with the functions and scripts and format their own data to test the top-down
and bottom-up models.

Author: Mark Wilber, UCSB

"""

# Various variables for
years = np.arange(2009, 2015)
sites = ['SiteA', 'SiteB', "SiteC"]
hosts = ["BUBO", "PSRE", "RACA", "TATO", "TAGR"] # hosts to include
parasites = ["GLOB", "MANO", "RION", "ECSP", "ALAR"] # parasites to include
splitby = "sitename" # SITE string
num_hosts = 10
num_parasites = 20

def expandgrid(*itrs, **kwargs):
   product = list(itertools.product(*itrs))
   return {kwargs['cnames'][i]:[x[i] for x in product] for i in range(len(itrs))}

# Get all combinations of vals
grid_vals = expandgrid(years, sites, hosts, range(20),
                cnames=["year", "sitename", "speciescode", "vals"])

dummy_dat = pd.DataFrame(grid_vals)

# Fill parasite columns with some random numbers. Just draw a bunch of Poisson
# RVS's
for para in parasites:
    dummy_dat[para] = stats.poisson.rvs(num_parasites / num_hosts,
                                    size=len(dummy_dat))

# Make dummy variable for svl
dummy_dat['svl'] = md.lognorm.rvs(1, 1, size=len(dummy_dat))

# Save dummy data
dummy_dat.to_csv("dummy_data.csv")


