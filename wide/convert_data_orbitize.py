import orbitize
import orbitize.read_input

from astropy.table import Table, Column
from astropy.io import ascii
from astropy.time import Time

# read the data ourselves and convert to orbitize file format
data = ascii.read("data/visual_data_besselian.csv", format="csv")

# convert Besselian year to MJD = JD - 2400000.5

t = Time(data["epoch"].data, format="byear")
data["epoch"] = t.mjd

data["sep"].name = "quant1"
data["sep_err"].name = "quant1_err"

data["pa"].name = "quant2"
data["pa_err"].name = "quant2_err"

# add a column for quant_type
data.add_column(Column(["seppa"] * len(data)), name="quant_type")

# write the data in the orbitize format
ascii.write(data, "data/visual_data.csv", overwrite=True)

print(orbitize.read_input.read_file("data/visual_data.csv"))
