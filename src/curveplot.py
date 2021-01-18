import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib as mp
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument("infile",type=str)
args = parser.parse_args()

mp.style.use('seaborn-colorblind')

def lcplot(t,f,color="k",line=".",lw=1):
    '''
    Simple lightcurve plot function
    '''
    mp.rcParams['figure.figsize'] = [12, 6]
    plt.plot(t,f,color+line,linewidth=lw, alpha = 0.1)
    plt.xlabel("Timestamp (days)")
    plt.ylabel(r"SAP flux ($e^-/s$)")
   
hdul = fits.open(args.infile)
lcdata = hdul[1].data
hdul.close()
pdcflux_raw = lcdata.field(7)
time_raw = lcdata.field(0)
has_errors = np.isnan(pdcflux_raw)
pdcflux = pdcflux_raw[~has_errors]
time = time_raw[~has_errors]

lcplot(time, pdcflux)
plt.show()
