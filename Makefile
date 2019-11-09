# set up the directory output structure for chains if it doesn't exist




.PHONY : close-rv-figs 
close-rv-figs : figures/close/rv/RV.pdf figures/close/rv/corner.png figures/close/rv/summary.txt figures/close/rv/trace.png 


# define typical outputs in each directory 
outputs = summary.txt trace.png corner.png autocorr.png 


# target rule to make all of the figures for a directory
.PHONY : rv_astro_less 
rv_astro_less : figures/close/rv_astro_less/RV.pdf figures/close/rv_astro_less/sep_pa.pdf

# each scenario has a subdirectory within src/ that contains model.py, sample.py, and plot.py
# sample.py creates chains 
# plot.py uses chains and creates figures

# create phony targets for all figures corresponding to these scenarios
### close/
# rv
# rv_astro_less
# rv_astro_more
# rv_astro_disk_less
# rv_astro_disk_more
### wide/
# astro 
# astro_rv_inc
# astro_rv_dec
### joint/
# rv_astro_inc
# rv_astro_dec
# rv_astro_disk_inc
# rv_astro_disk_dec


# every time the samples change or the plot script changes, we need to remake the plots
# (we don't need to remake it if samples doesn't exist, tho)
figures/%/RV.pdf : chains/%/0/samples.npz src/%/plot.py
	echo $<
	python src/$*/plot.py 

# every time the model.py or sample.py files are changed, we need to resample the model
# $* is the stem (in the recipe) matched by % (in the rule)
# wildcards `*` cannot appear in the target
chains/%/0/samples.npz : src/%/model.py src/%/sample.py
	python src/$*/sample.py

# even though the chains are intermediate files for plotting, don't delete them 
# e.g., intermediate products of implicit chains are deleted if they didn't exist 
# when make was invoked: 
# https://www.gnu.org/software/make/manual/html_node/Using-Implicit.html#Using-Implicit 
# https://www.gnu.org/software/make/manual/html_node/Chained-Rules.html
.PRECIOUS : chains/%/0/samples.npz
