# Bash script to run the full top-down and bottom-up analysis for
# feasible_manuscript.tex

# Notes
# -----
# The data used in this analysis is not yet distributable so this run_all
# script uses the dummy_data provides in data/archival/dummay_data.csv.  The
# full analysis takes approximately 1 hr on the dummy data.

echo "Running manuscript_analysis_no_hetero. See no_hetero.log for progress"
python manuscript_analysis_no_hetero.py
echo "Completed manuscript_analysis_no_hetero.py"

echo "Running manuscript_analysis_discrete_known_heterogeneity.py"
echo "See discrete_known_hetero.log for progress"
python manuscript_analysis_discrete_known_heterogeneity.py
echo "Completed manuscript_analysis_discrete_known_heterogeneity.py"

echo "Running manuscript_analysis_random_grouping.py. See randomization.log for progress"
python manuscript_analysis_random_grouping.py
echo "Completed manuscript_analysis_random_grouping.py"

echo "Running manuscript_analysis_continuous_hetero.py. See continuous_hetero.log for progress"
python manuscript_analysis_continuous_hetero.py
echo "Completed manuscript_analysis_continuous_hetero.py"

echo "Running manuscript_analysis_parasite_mortality.py."
echo "See parasite_mortality.log for progress"
python manuscript_analysis_parasite_mortality.py
echo "Completed manuscript_analysis_parasite_mortality.py"

echo "Running manuscript_analysis_no_hetero_bs_gof.py"
echo "See no_hetero_bs_gof.log for progress"
python manuscript_analysis_no_hetero_bs_gof.py
echo "Completed manuscript_analysis_no_hetero_bs_gof.py"

echo "Running manuscript_analysis_discrete_known_heterogeneity_aicc.py"
echo "See discrete_known_hetero_likelihood.log for progress"
python manuscript_analysis_discrete_known_heterogeneity_aicc.py
echo "Completed manuscript_analysis_discrete_known_heterogeneity_aicc.py"

echo "Completed analysis"
