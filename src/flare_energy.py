import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simps, quad
from scipy.optimize import curve_fit
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

# Returns the chi-squared value of the given data with predictions from the model

def chisq(data, model):
    """
    """    

    return np.sum((data-model)**2/model) 
    

def Model_exp(t, A, b):
    """
    Models a flare that starts from t0, has a peak A, and decays like
    F(t) = Ae^(-b(t-t0)). 
    """
    y = np.piecewise(
        t, [t < 0, t >= 0], [lambda tp: 0.0, lambda tp: A * np.exp(-b * tp)]
    )
    return y


def fit_exp(x, y):
    """
    Fits a simple exponential model to data
    """
    f = Model_exp

    # Initial guess roughly based on:
    #  - peak at maximum index
    #  - by flare's end, decay to 0.05 times the peak
    #  - start @ the beginning

    maxindex = np.argmax(y)

    guess = [1.0, -np.log(0.05) / (x[-FLARE_PAD_1] - x[maxindex])]

    popt, pcov = curve_fit(f, x, y, p0=guess, method="trf")

    Fres_model = f(x, *popt)
    return Fres_model


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


def zero(ts):
    return np.zeros(len(ts))


def Model_double_exp(t, A1, A2, k1, k2):
    """
    Models the tail of a flare that starts from t0, has a peak A, and decays like
    F(t) = A1*e^(-k1(t-t0)) + A2*e^(-k2(t-t0)) 
    """

    y = A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t)
    return y


def Model_quad_poly_fix(t, A, a2):
    a1 = a2 + A
    y = A + a1 * t + a2 * t ** 2
    return y


def Model_quart_poly_fix(t, A, a1, a3, a4):
    """
    Models the buildup of a flare that starts from t0, has a peak at x=0 (given by A), 
    and behaves like a quartic polynomial. Is also fixed to have derivative and value zero at
    x = -1
    """
    a2 = -a4 + a3 + a1 - A  # 2*a3 - 3*a4 + A
    # a1 = (4*a4 - 3*a3 + 2*a2)
    y = a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + A
    return y


def Model_quart_poly(t, a1, a2, a3, a4):
    """
    Models the buildup of a flare that starts from t0, has a peak at A, and behaves like
    a quartic polynomial.
    """
    y = a1 * t + a2 * t ** 2 + a3 * t ** 3 + a4 * t ** 4 + 1
    return y


def Model_Davenport(t):
    """
    Simple version of the Davenport model, specifically for fitting multiple flares. 
    """
    exp_params = [0.6890, 0.3030, 1.600, 0.2783]
    quart_params = [1.941, -0.175, -2.246, -1.125]

    y = np.zeros(len(t))
    y[t >= 0.0] = Model_double_exp(t[t >= 0.0], *exp_params)
    mask = ((t >= -1.) & (t < 0.0))
    y[mask] = Model_quart_poly(t[mask], *quart_params)
    
    return y
     

def fit_Davenport(x, y):
    """
    Fits Davenport's model from Davenport et al. 2014 [TODO citation here]
    """

    f1 = Model_double_exp
    # Initial guess: sharper exp higher, maximum at t0

    guess1 = [
        3.0 / 4.0,
        1.0 / 4.0,
        -np.log(0.05) / (x[-FLARE_PAD_1]),
        -np.log(0.1) / (x[-FLARE_PAD_1]),
    ]

    try:
        bounds = (0.0, [2.0, 2.0, np.infty, np.infty])

        popt1, pcov1 = curve_fit(
            f1, x[x >= 0], y[x >= 0], p0=guess1, method="dogbox", bounds=bounds
        )

        A = popt1[0] + popt1[1]

        f0 = lambda tp, a1, a3, a4: Model_quart_poly_fix(tp, A, a1, a3, a4)
        # f0 = lambda tp, a1, a2, a3, a4: Model_quart_poly(tp, A, a1, a2, a3, a4)
        mask0 = np.logical_and(x < 0, x >= -1)
        popt0, pcov0 = curve_fit(f0, x[mask0], y[mask0])
        Fres_model = np.zeros(len(x))
        Fres_model[mask0] = f0(x[mask0], *popt0)
        Fres_model[x >= 0] = f1(x[x >= 0], *popt1)

    except RuntimeError as err:
        Fres_model = fit_exp(x, y)
        print("{0}".format(err))

    return Fres_model



def fit_Davenport_2(x, y):
    """
    Fits Davenport's model from Davenport et al. 2014 [TODO citation here]
    """

    try:
        f = lambda xp, a, b : a * Model_Davenport(xp*b)
        popt1, pcov1 = curve_fit(
            f, x, y
        )

        Fres_model = popt1[0]*Model_Davenport(x*popt1[1])

    except RuntimeError as err:
        Fres_model = fit_exp(x, y)
        print("{0}".format(err))

    return Fres_model




def fit_multiple_flare(Fres, n, peaktimes, t, model="exp"):

    """
    Returns the optimized flare according to a model, and its energy.
    The model should also have the first parameter correspond to the peak of
    the flare. The model is fit using an interpolation of the flux residuals,
    by default normalized to the units used in Davenport (2014);
    some helper values are provided in the beginning.
    """

    time0 = time()
    interpol = interp1d(t, Fres, fill_value="extrapolate")
    x = np.arange(t[0], t[-1], 10.0) # Unit is seconds here!
    y = interpol(x)
    initialguess = []
    y_model = np.zeros(len(y))
    
    maxx = x[np.argmax(np.abs(y))]
    maxy = np.max(y)
    thalf = find_thalf(x, y, maxx, maxy)  # TODO DAVENPORT
    x = (x - maxx) / thalf
    y = y / maxy

    if model == "exp":
        f = Model_exp

    if model == "Davenport" or model == "Davenport2":
        f = Model_Davenport

    for i in range(n):
        # Find an initial guess
        y_i = y - y_model
        fp = lambda tp, a, b : a*f(b*(tp - peaktimes[i]))
        popt, pcov = curve_fit(fp, x, y_i)
        initialguess.extend(popt)

        fi = lambda tp, *args: np.sum(
            [args[2 * j] * f(args[2 * j + 1]*(tp - peaktimes[j])) for j in range(i + 1)],axis=0
        )

        # Dirty one liner that sums up the flares for an i-flare model. Currently only works for
        # a 2 parameter model f.

        popt, pcov = curve_fit(fi, x, y, p0=initialguess)
        for j in range(len(initialguess)):
            initialguess[j] = popt[j]

        y_model = fi(x, *popt)
        
    
    times = x * thalf + maxx

    # TODO BIC CALCULATION HERE
    
    if PLOT == True:
        plt.plot(x, y_model*maxy)
        for i in range(n):
            Fmodel_i = popt[2 * i]*f(popt[2 * i + 1]*(x-peaktimes[i]))*maxy
            plt.plot(x, Fmodel_i)
        plt.plot((t-maxx)/thalf, Fres, "ro")
        plt.show(block = False)

    Fres_models = []
    Fres_peaks = []
    t_peaks = []

    for i in range(n):
        Fres_models.append(popt[2 * i]*f(popt[2 * i + 1]*(x-peaktimes[i]))*maxy)
        Fres_peaks.append(np.max(Fres_models[i]))
        t_peaks.append(times[np.argmax(Fres_models[i])])

    return (Fres_peaks, times, Fres_models, t_peaks)


def Model_flare(Fres, t, model="exp"):

    """
    Returns the optimized flare according to a model, and its energy.
    The model should also have the first parameter correspond to the peak of
    the flare. The model is fit using an interpolation of the flux residuals,
    by default normalized to the units used in Davenport (2014);
    some helper values are provided in the beginning.
    """
    
    time0 = time()
    interpol = interp1d(t, Fres, fill_value="extrapolate")
    x = np.arange(t[0], t[-1], 10.0)
    y = interpol(x)
    maxx = x[np.argmax(y)]
    maxy = np.max(y)
    thalf = find_thalf(x, y, maxx, maxy)
    x = (x - maxx) / thalf
    y = y / maxy

    Fres_model = np.zeros(len(x))

    if model == "exp":
        Fres_model = fit_exp(x, y) * maxy

    if model == "Davenport":
        Fres_model = fit_Davenport(x, y) * maxy

    if model == "Davenport2":
        Fres_model = fit_Davenport_2(x, y) * maxy

    times = x * thalf + maxx
    xt = (t - maxx)/thalf

    if PLOT == True:
        plt.plot(x, Fres_model)
        plt.plot(xt, Fres, "ro")
        plt.show(block = False)

    return (maxy, times, Fres_model)


def Response_TESS():

    """
    Returns the TESS response curve as a function of the wavelength
    lambd. It's a spline interpolation 
    of the TESS response function.
    """

    responsefile = "src/tess-response-function-v1.0.csv"
    rs = np.genfromtxt(responsefile, delimiter=",")
    f = interp1d(rs[:, 0], rs[:, 1])
    return f


R_tess = Response_TESS()


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


def Lp_FLARE_DENSITY(T_flare, R_TESS):

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


def main():

    parser = ap.ArgumentParser(
        description="Estimate flare energies from a lightcurve with flagged flare times"
    )
    parser.add_argument("inputfile", metavar="if", type=str, help="Input file")
    parser.add_argument("starinfo", type=str, help="")
    args = parser.parse_args()

    LpS = Lp_STAR(R_STAR, T_STAR, R_tess)
    LpF = Lp_FLARE_DENSITY(T_FLARE, R_tess)

    # Load the lightcurve with flagged flare times

    data = np.loadtxt(args.inputfile)
    pdcflux = data[:, 1]
    times = data[:, 0]
    trend = data[:, 2]
    flarestamps = data[:, 3]
    pdcflux_ratio = (pdcflux - trend) / trend

    flare_energies = []
    flare_impulses = []
    flare_times = []
    in_flare = False
    flare_dur = 0

    for i in range(len(pdcflux)):
        if np.isnan(trend[i]):
            if in_flare:
                flare_dur += 1
                continue

        if flarestamps[i] == 1:
            in_flare = True
            flare_dur += 1

        elif in_flare:
            amps = pdcflux_ratio[i - flare_dur - FLARE_PAD_1 : i + FLARE_PAD_2]
            ts = times[i - flare_dur - FLARE_PAD_1 : i + FLARE_PAD_2] * DAY
            nans = np.isnan(amps)
            Fres = amps[~nans]
            t = ts[~nans]

            Fmodel = Model_flare(Fres, t, model=FLARE_MODEL)
            
            # BIC = chisq(Fres, Fmodel[2]) + np.log(len(Fres))
            # print(BIC)
            # TODO BIC will still need the parametric model for the length of t/Fres/whatever

            multiflare = "n"
            retry = "y"

            accept = input("Accept model? Y/N\n")
            
            if accept == "N" or accept == "n":
                multiflare = input("Multiple flare model? Y/N\n")
                if multiflare == "n" or multiflare == "N":
                    print("Flare rejected.")
            else:
                (i_f, E_f) = E_flare(Fmodel, T_FLARE, LpS, LpF, R_STAR)
                flare_energies.append(E_f / ERG)
                flare_impulses.append(i_f)
                flare_times.append(times[i - flare_dur])

            while(True):
                if multiflare == "n" or multiflare == "N":
                    break
                if retry == "n" or retry == "N":
                    break

                nflare = int(input("Number of flares:\n"))

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

                Fmodel = fit_multiple_flare(Fres, nflare, tpeak, t, model=FLARE_MODEL)
                
                Fmodel_sum = np.sum(Fmodel[2])
                # TODO implement getting Fres as a model
                # BIC = nflare * np.log(len(Fmodel_sum)) + chisq(Fres, Fmodel)
                # print(BIC)

                accept = input("Accept model? Y/N\n")
                if (accept == "n" or accept == "N"):
                    retry = input("Retry fit? Y/N\n")
                    continue

                for i in range(nflare):
                    (i_f, E_f) = E_flare(
                        [Fmodel[0][i], Fmodel[1], Fmodel[2][i]], T_FLARE, LpS, LpF, R_STAR
                    )
                    flare_energies.append(E_f / ERG)
                    flare_impulses.append(i_f)
                    flare_times.append(Fmodel[3][i] / DAY)
                
                break
                           
            flare_dur = 0
            in_flare = False
            

    # Print the flare energies to stdout
    
    output = []

    for i in range(len(flare_energies)):
        output.append(
            "{:.6f}  {:.8e}  {:.8e}\n".format(
                flare_times[i], flare_energies[i], flare_impulses[i]
            )
        )
    
    outf = open("out/energies.out", "a")
    outf.writelines(output)
    outf.close()
    
if __name__ == "__main__":
    main()
