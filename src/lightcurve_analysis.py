from astropy.io import fits
import argparse as ap
import numpy as np
from sklearn import svm
from scipy import integrate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from copy import *


params = np.loadtxt("src/params.dat")
PERIOD = params[0]
WINDOWPERIOD = params[1]
WINDOW_OVERLAP = params[2]
PEAK_SENS = params[3]
ROLLOFF_SENS = params[4]
MIN_DURATION = params[5]


def detrended_curve(time, flux):

    """    
    Fits a support vector machine to the curve (time, flux defined)
    with radial kernel functions; returns the trended curve.
    requires the line "from sklearn import svm"
    """
    
    if len(time) == 0:
        return []

    meant = np.mean(time)
    stdt = np.std(time)
    meanf = np.mean(flux)
    stdf = np.std(flux)

    wt = [[(t - meant) / stdt] for t in time]  # normalization for optimal SVM
    clf = svm.SVR(kernel="rbf", gamma="auto")
    try:
        clf.fit(wt, (flux - meanf) / stdf)
    except ValueError:
        return []
        
    flux_pred = clf.predict(wt)

    return flux_pred * stdf + meanf


def flare_cand_spot(dt_flux, sigma):

    """
	Returns the indices of the flare candidates within a window.
	Ignores nans.
	Goes through jumps, and tags them if they appear significant.
	"""

    flare_times = np.zeros(len(dt_flux))
    len_flare = 0
    df = [dt_flux[i + 1] - dt_flux[i] for i in range(len(dt_flux) - 1)]

    # Tag the points where we spot the flare getting over 2 sigma;
    # two consecutive points will suffice:

    for i in range(len(dt_flux)):
        if dt_flux[i] > PEAK_SENS * sigma:
            len_flare += 1
        elif len_flare > MIN_DURATION and dt_flux[i] > ROLLOFF_SENS * sigma:
            len_flare += 1
        elif len_flare > MIN_DURATION:
            flare_times[i - len_flare : i] = 1.0
            len_flare = 0
        else:
            len_flare = 0

    return flare_times


def trend_lightcurve(time_raw, pdcflux_raw, windowsize, windowstep):

    """
    the trend of the lightcurve - up to 7 overlapping windows fit the curve with
    svm, take the median of them
    includes NaNs!
    """

    flux_raw = deepcopy(pdcflux_raw)

    n_votes = np.zeros(len(time_raw))
    lightcurve_trended = np.zeros(len(time_raw))
    flare_points = np.zeros(len(time_raw))

    for i in range(0, len(time_raw), windowstep):
        iend = min(len(time_raw), i + windowsize)
        windowflux_raw = deepcopy(flux_raw[i:iend])
        windowtime_raw = time_raw[i:iend]
        has_errs = np.isnan(windowflux_raw)
        flux = windowflux_raw[~has_errs]
        time = windowtime_raw[~has_errs]
        flux_trend = detrended_curve(time, flux)
        if (len(flux_trend) == 0): continue
        
        windowflux_raw[~has_errs] = flux_trend
        lightcurve_trended[i : i + windowsize] += windowflux_raw
        n_votes[i : i + windowsize] += 1
    
    lightcurve_trended[lightcurve_trended == 0] = np.nan
    lightcurve_trended[n_votes != 0] /= n_votes[n_votes != 0]
    lightcurve_detrended = pdcflux_raw - lightcurve_trended
    errs = np.isnan(lightcurve_detrended)
    sigma = np.std(lightcurve_detrended[~errs])

    for i in range(0, len(time_raw) - windowsize, windowstep):
        
        flare_points[i : i + windowsize] = flare_cand_spot(
            lightcurve_detrended[i : i + windowsize], sigma
        )

    return lightcurve_trended, flare_points, sigma


def lcplot(t, f, color=""):
    """
    Simple lightcurve plot function
    """
    plt.plot(t, f, color + ".", alpha=0.1)
    plt.xlabel("Timestamp (days)")
    plt.ylabel(r"SAP flux ($e^-/s$)")


parser = ap.ArgumentParser(description="Trend a lightcurve from a TESS FITS file")
parser.add_argument("inputfile", metavar="if", type=str, help="Input file")
#parser.add_argument("period", type=float, help="Period of the star")
parser.add_argument("--plot", help="Plot the lightcurve in file", type=str)
parser.add_argument("--show", help="Show the lightcurve", action="store_true")
parser.add_argument("--of", type=str, help="Output file for data")

args = parser.parse_args()

hdul = fits.open(args.inputfile)
lcdata = hdul[1].data
pdcflux_raw = lcdata.field(7)
time_raw = lcdata.field(0)
hdul.close()

cadence = len(time_raw) / (time_raw[len(time_raw) - 1] - time_raw[0])

windowsize = int(WINDOWPERIOD * PERIOD * cadence)
windowstep = int(windowsize / WINDOW_OVERLAP)

lc_trend, flares, sigma = trend_lightcurve(
    time_raw, pdcflux_raw, windowsize, windowstep
)

for i in range(len(time_raw)):
    print("{} {} {} {}".format(time_raw[i], pdcflux_raw[i], lc_trend[i], flares[i]))

if args.plot:
    plotname = args.plot
    lcplot(time_raw, pdcflux_raw)
    plt.plot(time_raw[flares == 1.0], pdcflux_raw[flares == 1.0], ".", alpha=0.5)
    plt.plot(time_raw, lc_trend, "r--")
    plt.savefig(plotname)
    if args.show:
        plt.show()
