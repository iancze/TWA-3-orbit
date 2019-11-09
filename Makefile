# each scenario has a subdirectory within src/ that contains model.py, sample.py, and plot.py
# sample.py creates chains 
# plot.py uses chains and creates figures

# define typical outputs in each directory 
outputs = summary.txt trace.png corner.png autocorr.png 

# some of these directories will also have PDFs of the radial velocity curve
# and astrometric predictions

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

# target rule to make all of the figures for a directory

.PHONY : close-rv
close-rv : $(addprefix figures/close/rv/,$(outputs))

.PHONY : close-rv_astro_less 
close-rv_astro_less : $(addprefix figures/close/rv_astro_less/,$(outputs))

.PHONY : close-rv_astro_more
close-rv_astro_more : $(addprefix figures/close/rv_astro_more/,$(outputs))

.PHONY : close-rv_astro_disk_less 
close-rv_astro_disk_less : $(addprefix figures/close/rv_astro_disk_less/,$(outputs))

.PHONY : close-rv_astro_disk_more
close-rv_astro_disk_more : $(addprefix figures/close/rv_astro_disk_more/,$(outputs))

.PHONY : wide-astro 
wide-astro : $(addprefix figures/wide/astro/,$(outputs))

.PHONY : wide-astro_rv_inc
wide-astro_rv_inc : $(addprefix figures/wide/astro_rv_inc/,$(outputs))

.PHONY : wide-astro_rv_dec
wide-astro_rv_dec : $(addprefix figures/wide/astro_rv_dec/,$(outputs))



# this expression expands out to the sample directories
# we assume we are always running 4 chains.
# $(foreach var,0 1 2 3,chains/%/$(var)/samples.npz)

# every time the samples change or the plot script changes, we need to remake the plots
$(addprefix figures/%/,$(outputs)) : $(foreach var,0 1 2 3,chains/%/$(var)/samples.npz) src/%/plot.py 
	python src/$*/plot.py 

# every time the model.py or sample.py files are changed, we need to resample the model
# $* is the stem (in the recipe) matched by % (in the rule)
# wildcards `*` cannot appear in the target
$(foreach var,0 1 2 3,chains/%/$(var)/samples.npz) : src/%/model.py src/%/sample.py
	python src/$*/sample.py

# even though the chains are intermediate files for plotting, don't delete them 
# e.g., intermediate products of implicit chains are deleted if they didn't exist 
# when make was invoked: 
# https://www.gnu.org/software/make/manual/html_node/Using-Implicit.html#Using-Implicit 
# https://www.gnu.org/software/make/manual/html_node/Chained-Rules.html
.PRECIOUS : $(foreach var,0 1 2 3,chains/%/$(var)/samples.npz)