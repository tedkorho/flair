import argparse as ap
import numpy as np
from sklearn import svm
from scipy import integrate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from copy import *

K_B = 1.3806485e-23
C_PLANCK = 6.62607004e-34
C_C = 299792458.0
SIGMA_SB = 5.670374419e-8
NANOMETER = 1.0e-9
LAMBDA_MIN = 454.06
LAMBDA_MAX = 1129.15

params = np.loadtxt("src/params.dat")

PERIOD = params[0]
WINDOWPERIOD = params[1]
WINDOW_OVERLAP = params[2]
PEAK_SENS = params[6]
ROLLOFF_SENS = params[7]
MIN_DURATION = params[8]


def detrended_curve(time, time0, flux0):

    """    
    Fits a support vector machine to the curve (time, flux defined)
    with radial kernel functions; returns the trended curve.
    requires the line "from sklearn import svm"
	time is fo
    """

    meant = np.mean(time0)
    stdt = np.std(time0)
    meanf = np.mean(flux0)
    stdf = np.std(flux0)

    wt0 = [[(t - meant) / stdt] for t in time0]  # normalization for optimal SVM
    clf = svm.SVR(kernel="rbf", gamma="auto")
    clf.fit(wt0, (flux0 - meanf) / stdf)
    wt = [[(t - meant) / stdt] for t in time]
    flux_pred = clf.predict(wt)

    return flux_pred * stdf + meanf


def flare_spot(dt_flux, sigma):

    """
	Returns the indices of the flares within a window.
	Ignores nans.
	Goes through jumps, and tags them if
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


def retrend_lightcurve(time_raw, pdcflux_raw, flares, windowsize, windowstep):

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
        windowflares = deepcopy(flares[i:iend])
        windowtime_raw = time_raw[i:iend]

        has_errs = np.isnan(windowflux_raw)
        flux0 = windowflux_raw[~has_errs]
        time0 = windowtime_raw[~has_errs]
        flare0 = windowflares[~has_errs]

        flux1 = flux0[flare0 != 1.0]
        time1 = time0[flare0 != 1.0]

        if len(time1) < 2:
            continue

        flux_trend = detrended_curve(time0, time1, flux1)
        windowflux_raw[~has_errs] = flux_trend
        lightcurve_trended[i : i + windowsize] += windowflux_raw
        n_votes[i : i + windowsize] += 1

    lightcurve_trended[np.where(lightcurve_trended == 0)[0]] = np.nan
    lightcurve_trended /= n_votes
    lightcurve_detrended = pdcflux_raw - lightcurve_trended
    errs = np.isnan(lightcurve_detrended)
    sigma = np.std(lightcurve_detrended[~errs])

    for i in range(0, len(time_raw) - windowsize, windowstep):

        flare_points[i : i + windowsize] = flare_spot(
            lightcurve_detrended[i : i + windowsize], sigma
        )

    return lightcurve_trended, flare_points


def lcplot(t, f, color=""):
    """
    Simple lightcurve plot function
    """
    plt.plot(t, f, color + ".", alpha=0.1)
    plt.xlabel("Timestamp (days)")
    plt.ylabel(r"SAP flux ($e^-/s$)")


parser = ap.ArgumentParser(
    description="Re-trend a lightcurve from an output file by the trend_lightcurves script"
)
parser.add_argument("inputfile", metavar="if", type=str, help="Input file")
# parser.add_argument("period", type=float, help="Period of the star")
parser.add_argument("--plot", help="Plot the lightcurve in file", type=str)
parser.add_argument("--show", help="Show the lightcurve", action="store_true")
parser.add_argument("--of", type=str, help="Output file for data")

args = parser.parse_args()

lcdata = np.loadtxt(args.inputfile)
time = lcdata[:, 0]
pdcflux = lcdata[:, 1]

# also prepare for the case where the input data does not have flare stamps


if len(lcdata[0]) > 3:
    flares0 = lcdata[:, 3]
else:
    flares0 = np.zeros(len(pdcflux))


# hdul = fits.open(args.inputfile)
# lcdata = hdul[1].data
# pdcflux_raw = lcdata.field(7)
# time_raw = lcdata.field(0)
# hdul.close()

cadence = len(time) / (time[len(time) - 1] - time[0])

windowsize = int(WINDOWPERIOD * PERIOD * cadence)
windowstep = int(windowsize / WINDOW_OVERLAP)

lc_trend, flares = retrend_lightcurve(time, pdcflux, flares0, windowsize, windowstep)

for i in range(len(time)):
    print("{} {} {} {}".format(time[i], pdcflux[i], lc_trend[i], flares[i]))

if args.plot:
    plotname = args.plot
    lcplot(time, pdcflux)
    plt.plot(time[flares == 1.0], pdcflux[flares == 1.0], ".", alpha=0.5)
    plt.plot(time, lc_trend, "r--")
    plt.savefig(plotname)
    if args.show:
        plt.show()
