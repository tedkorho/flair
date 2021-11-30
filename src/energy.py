import numpy as np
from numpy.polynomial import Polynomial
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simps, quad
from scipy.optimize import curve_fit, minimize
from sklearn import svm
from astropy.io import fits
import argparse as ap
from time import time

K_B = 1.3806485e-23
C_PLANCK = 6.62607004e-34
C_C = 2.99792458e8
SIGMA_SB = 5.670374419e-8
LAMBDA_MIN = 454.06
LAMBDA_MAX = 1129.15
NANOMETER = 1.0e-9
ERG = 1.0e-7
R_SUN = 696340000.0
DAY = 86400

params = np.loadtxt("src/params.dat")

R_STAR = params[9] * R_SUN
T_STAR = params[10]
T_FLARE = params[11]
FM_NUM = int(params[12])
FLARE_MODEL = "exp"
if FM_NUM == 2:
    FLARE_MODEL = "Davenport"
elif FM_NUM == 3:
    FLARE_MODEL = "Davenport2"

FLARE_PAD_1 = int(params[13])
FLARE_PAD_2 = int(params[14])
PLOT = True

# TODO REFACTORING:
#    - more uniform variable naming! (go through this w/ notebook)
#    - find & minimize hardcoded constants 
#    - consider:
#       some functions in a separate file
#       a clear place to add flare models

# TODO Issues to solve (then all done):
#    - the thing where the base level comes from a polynomial fit - BASICALLY DONE
#    - BIC working properly - MOSTLY DONE
#    - ensure that the energy is integrated correctly: easiest done by comparing
#       to the AB Dor paper - MOSTLY DONE

def svr_trend(t, fres):
    """
    Fits a SVR model for the time before & after the flare; works as a reference level.
    """
    
    pre = int(FLARE_PAD_1/2)
    post = int(FLARE_PAD_2/2)

    tfit = np.zeros(pre+post)
    tfit[:pre] = t[:pre]
    tfit[-post:] = t[-post:]
    ffit = np.zeros(pre+post)
    ffit[:pre] = fres[:pre]
    ffit[-post:] = fres[-post:]
    meant = np.mean(tfit)
    stdt = np.std(tfit)
    meanf = np.mean(ffit)
    stdf = np.std(ffit)

    wt = np.array([[(tp - meant) / stdt] for tp in tfit])  # normalization for optimal SVM
    clf = svm.SVR(kernel="rbf", gamma="auto")
    try:
        clf.fit(wt, (ffit - meanf) / stdf)
    except ValueError:
        return np.array([])

    flux_pred = clf.predict(np.array([[(tp - meant)/stdt] for tp in t]))

    return flux_pred * stdf + meanf


def mse(data, model):
    """
    Is supposed to return the chi-squared (or similar) value indicating
    the error in the model
    """

    return np.sum((data - model) ** 2)/len(data)


def model_exp(t, A, b):
    """
    Models a flare that starts from t0, has a peak A, and decays like
    F(t) = Ae^(-b(t-t0)). 
    """
    y = np.piecewise(
        t, [t < 0, t >= 0], [lambda tp: 0.0, lambda tp: A * np.exp(-b * tp)]
    )
    return y


def find_thalf(x, y, maxx, maxy):
    """
    Helper function to find the characteristic time used to normalize
    a fit
    """
    ctr = 0
    t1 = x[0]
    t2 = maxx
    for i in range(len(x)):
        if ctr == 0 and y[i] > 0.5 * maxy:
            ctr = 1
            t1 = x[i]
        if ctr == 1 and y[i] < 0.5 * maxy:
            t2 = x[i]
            break

    thalf = t2 - t1
    return thalf


def model_double_exp(t, A1, A2, k1, k2):
    """
    Models the tail of a flare that starts from t0, has a peak A, and decays like
    F(t) = A1*e^(-k1(t-t0)) + A2*e^(-k2(t-t0)) 
    """

    y = A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t)
    return y


def model_quad_poly_fix(t, A, a2):
    a1 = a2 + A
    y = A + a1 * t + a2 * t ** 2
    return y


def model_quart_poly_fix(t, A, a1, a3, a4):
    """
    Models the buildup of a flare that starts from t0, has a peak at x=0 (given by A), 
    and behaves like a quartic polynomial. Is also fixed to have derivative and value zero at
    x = -1
    """
    a2 = -a4 + a3 + a1 - A
    y = a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + A
    return y


def model_quart_poly(t, a1, a2, a3, a4):
    """
    Models the buildup of a flare that starts from t0, has a peak at A, and behaves like
    a quartic polynomial.
    """
    y = a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + 1
    return y


def model_davenport(t):
    """
    Simple version of the Davenport model, specifically for fitting multiple flares. 
    """
    exp_params = [0.6890, 0.3030, 1.600, 0.2783]
    quart_params = [1.941, -0.175, -2.246, -1.125]

    y = np.zeros(len(t))
    y[t >= 0.0] = model_double_exp(t[t >= 0.0], *exp_params)
    mask = (t >= -1.0) & (t < 0.0)
    y[mask] = model_quart_poly(t[mask], *quart_params)

    return y


def tikhonov_error(f, x, y, params, alpha):
    """
    Returns a Tikhonov style error. ||f(x, *params) - y|| + alpha||params||
    x, y are 1D arrays of the same dimension n.
    params is an arbitrary length array.
    f is a function such that f(x, *params) returns an 1D array of dimension n.
    alpha is a scalar
    """

    return np.linalg.norm(f(x, *params) - y) + alpha*np.linalg.norm(params)




def plot_flare_models(tdata, fres_data, tmodel, ymodel, peaktimes, f, popt, BIC, nflare):
    """
    Plots flares
    """

    plt.clf()
    plt.plot(tmodel, ymodel)
    plt.legend(["BIC = {:.3f}".format(BIC)])
    nparam = int(len(popt)/nflare)
    for i in range(nflare):
        fmodel_i = (
            f(tmodel - peaktimes[i], *popt[nparam*i : nparam*(i+1)])
        )
        plt.plot(tmodel, fmodel_i)
    
    plt.plot(tdata, fres_data, "ro")
    plt.show(block=False)


def model_flare(fres, peaktimes, t, model="exp"):

    """
    Returns the optimized flare according to a model, and its energy.
    The model should also have the first parameter correspond to the peak of
    the flare. The model is fit using an interpolation of the flux residuals,
    by default normalized to the units used in Davenport (2014);
    some helper values are provided in the beginning.
    """

    n = len(peaktimes)

    time0 = time()
    interpol = interp1d(t, fres, fill_value="extrapolate")
    x = np.arange(t[0], t[-1], 10.0)  # Unit is seconds here!
    y = interpol(x)
    initialguess = []
    bounds = []

    y_model = np.zeros(len(y))
    y_obs = np.zeros(len(t))

    maxx = x[np.argmax(np.abs(y))]
    maxy = np.max(y)
    thalf = find_thalf(x, y, maxx, maxy)  # TODO DAVENPORT
    x = (x - maxx) / thalf
    y = y / maxy
    
    alpha = 0.0*len(x)

    if model == "exp":
        f = model_exp
        f0 = lambda x, a, b, c : a * f(b * (x - c))
        bounds_single = [(0.0,3.0),(0.0,10.0)]
        guess_single = [1., 1., 0.]
        nargs = 3

    if model == "Davenport" or model == "Davenport2":
        f = model_davenport
        f0 = lambda x, a, b, c : a * f(b * (x - c))
        bounds_single = [(0.0,3.0),(0.0,5.0),(-0.2,0.2)]
        bounds_alt = ([0.0, 0.0, -0.5],[3.0,5.0, 0.5])
        guess_single = [1., 0.5, 0.]
        nargs = 3

    for i in range(n):
        # Find an initial guess
        popt,pcov = curve_fit(f0, x, y-y_model, bounds = bounds_alt)
        initialguess.extend(popt)
        
        bounds.extend(bounds_single)

        fi = lambda tp, *args: np.sum(
            [
                f0(tp - peaktimes[j], *args[nargs*j : nargs*(j+1)])
                for j in range(i + 1)
            ],
            axis=0,
        )

        # Dirty one liner that sums up the flares for an i-flare model.

        error = lambda b : tikhonov_error(fi, x, y, b, alpha)
        opt = minimize(error, initialguess, bounds = bounds)

        for j in range(len(initialguess)):
            initialguess[j] = opt.x[j]

        y_model = fi(x, *opt.x)
        y_obs = fi((t - maxx)/thalf, *opt.x)*maxy

    times = x * thalf + maxx

    BIC = 2*np.log(mse(fres/maxy, y_obs/maxy))*len(fres) + nargs*n*np.log(len(fres))
    
    if PLOT == True:
        tdata = (t-maxx)/thalf
        plot_flare_models(tdata, fres/maxy, x, y_model, peaktimes, f0, opt.x, BIC, n)
        
    fres_models = []
    fres_peaks = []
    t_peaks = []

    for i in range(n):
        fres_models.append(
            opt.x[3 * i]
            * f(opt.x[3 * i + 1] * (x - peaktimes[i] - opt.x[3 * i + 2]))
            * maxy
        )
        fres_peaks.append(np.max(fres_models[i])*maxy)
        t_peaks.append(times[np.argmax(fres_models[i])])

    return (fres_peaks, times, fres_models, t_peaks)



def response_TESS():

    """
    Returns the TESS response curve as a function of the wavelength
    lambd. It's a spline interpolation 
    of the TESS response function.
    """

    responsefile = "src/tess-response-function-v1.0.csv"
    rs = np.genfromtxt(responsefile, delimiter=",")
    f = interp1d(rs[:, 0], rs[:, 1])
    return f




def B(lambd, T):

    """
    The Planck distribution as a function of wavelength (in nanometers)
    Returns the spectral energy density rather than spectral density.
    """

    l = lambd * NANOMETER
    denom = np.exp(C_PLANCK * C_C / (l * K_B * T)) - 1.0
    num = 2.0 * C_PLANCK * C_C ** 2.0 * l ** (-5.0)
    return num / denom


def L(lambd):
    return B(lambd, T) * R_tess(lambd)


def Lp_STAR(R_star, T_star, R_TESS):

    """
    Incident luminosity of the star in the wavelength band
    """

    integrand = lambda lam: B(lam, T_star) * R_TESS(lam)
    r = np.arange(LAMBDA_MIN, LAMBDA_MAX)
    integr = simps(integrand(r), r) * NANOMETER

    return (
        np.pi ** 2 * R_star ** 2 * integr
    )  # radiance -> flux density on a sphere -> luminosity


def Lp_flare_density(T_flare, R_TESS):

    """
    Incident luminosity density of the flare in the TESS band, divided by its
    area
    """

    integrand = lambda lam: B(lam, T_flare) * R_TESS(lam)
    r = np.arange(LAMBDA_MIN, LAMBDA_MAX)
    integr = simps(integrand(r), r) * NANOMETER

    return integr


def A_flare(rel_amp, LpS, LpF, R_STAR):

    """
    Returns the estimated effective area of the flare, given the
    relative amplitude from the normalized lightcurve;
    the luminosity of the star, and the luminosity of the flare per unit area
    """

    return LpS * rel_amp / LpF  # *R_STAR


def L_flare(T_flare, A_f):

    """
    The luminosity of the flare at a given time:
    takes in the area of the flare and its temperature
    """

    return SIGMA_SB * T_flare ** 4.0 * A_f


def E_flare(Fmodel, T_flare, LpS, LpF, R_star):

    """
    The energy of the flare, following a model. 
    Takes in the amplitude,
    and pre-calculated values for the luminosity of the 
    star and the luminosity of the flare per unit area.
    """

    L_f = lambda amp: L_flare(T_flare, A_flare(amp, LpS, LpF, R_star))

    # Fmodel is a tuple; [0] is the maximum,
    # [1] is the interpolated times,
    # [2] is the modelled flare as an array.

    impulsiveness = Fmodel[0] / 2
    lumin = L_f(Fmodel[2])
    energy = simps(lumin, Fmodel[1])

    return (impulsiveness, energy)

def input_int(prompt):
    """
    Prompts an integer from the user w/ error handling.
    """
    i = 0
    while(true):
        try:
            i = int(input(prompt))
            break
            
        except ValueError:
            print("Invalid input. Try again.")
            continue
    return i
    
def input_float(prompt):
    """
    Prompts a float from the user w/ error handling.
    """
    f = 0.
    while(true):
        try:
            f = float(input(prompt))
            break
            
        except ValueError:
            print("Invalid input. Try again.")
            continue
    return f

def input_bool(prompt):
    """
    Prompts a yes/no from the user w/ error handling.
    """
    answer = ""
    while(true):
        try:
            answer = input(prompt)
            break
            
        except ValueError:
            print("Invalid input. Try again.")
            continue
    
    if (answer == "N" or answer == "n"):
        return False
    else:
        return True


def process_candidate(t, flux):
    """
    Function to process a flare candidate.
    """
    fres = flux - svr_trend(t, flux)

    nflare = 1
    fmodel = model_flare(fres, [0.0], t, model=FLARE_MODEL)

    multiflare = "n"
    retry = "y"

    accept = input("Accept model? Y/N\n")

    if accept == "N" or accept == "n":
        multiflare = input("Multiple flare model? Y/N\n") #TODO input functions instead of try/except
        if multiflare == "n" or multiflare == "N":
            print("Flare rejected.")
    
    while True:
        if multiflare == "n" or multiflare == "N":
            break
        if retry == "n" or retry == "N":
            break
        
        try:
            nflare = int(input("Number of flares:\n"))
        except ValueError:
            continue

        tpeak = np.zeros(nflare)
        for i in range(nflare):
            ordstr = ""
            if i == 1:
                ordstr = " 2nd"
            elif i == 2:
                ordstr = " 3rd"
            elif i > 2:
                ordstr = "{:d}st".format(i)

            tpeak[i] = float(
                input("Time of{:s} highest flare:\n".format(ordstr))
            )

        fmodel = model_flare(fres, tpeak, t, model=FLARE_MODEL)

        accept = input("Accept model? Y/N\n")
        if accept == "n" or accept == "N":
            retry = input("Retry fit? Y/N\n")
            continue
        else:
            break
    
    ret = []

    for i in range(nflare):
        (i_f, E_f) = E_flare(
            [fmodel[0][i], fmodel[1], fmodel[2][i]/np.nanmean(flux)],
            T_FLARE,
            LpS,
            LpF,
            R_STAR,
        )
        
        ret.append(
            "{:.6f}  {:.8e}  {:.8e}\n".format(
                fmodel[3][i] / DAY, E_f / ERG, i_f
            )
        )

    return ret
    
R_tess = response_TESS()
LpS = Lp_STAR(R_STAR, T_STAR, R_tess)
LpF = Lp_flare_density(T_FLARE, R_tess)

def main():

    parser = ap.ArgumentParser(
        description="Estimate flare energies from a lightcurve with flagged flare times"
    )
    parser.add_argument("inputfile", metavar="if", type=str, help="Input file")
    parser.add_argument("starinfo", type=str, help="")
    args = parser.parse_args()

    # Load the lightcurve with flagged flare times

    data = np.loadtxt(args.inputfile)
    flux = data[:, 1]
    t = data[:, 0]
    trend = data[:, 2]
    flarestamps = data[:, 3]
    
    nans = np.isnan(flux)
    flux = flux[~nans]
    t = t[~nans]
    trend = trend[~nans]
    flarestamps = flarestamps[~nans]

    in_flare = False
    flare_dur = 0
    output = []

    for i in range(len(flux)):
        if np.isnan(trend[i]):
            if in_flare:
                flare_dur += 1
                continue

        if flarestamps[i] == 1:
            in_flare = True
            flare_dur += 1

        elif in_flare:
            win = np.s_[i - flare_dur - FLARE_PAD_1 : i + FLARE_PAD_2]
            twin = t[win] * DAY
            fwin = flux[win]
            output.extend(process_candidate(twin, fwin))
            flare_dur = 0
            in_flare = False

    # Print the flare energies to stdout

    outf = open("out/energies.out", "a")
    outf.writelines(output)
    outf.close()


if __name__ == "__main__":
    main()
