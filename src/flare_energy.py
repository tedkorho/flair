import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import simps, quad
from scipy.optimize import curve_fit
from astropy.io import fits
import argparse as ap

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

FLARE_PAD = 5  # TODO debating where to put this; it's how many timesteps
# the flare should be padded in each direction

params = np.loadtxt("src/params.dat")

R_STAR = params[9] * R_SUN
T_STAR = params[10]
T_FLARE = params[11]


def Model_exp(t, A, b, t0):
    """
    Models a flare that starts from t0, has a peak A, and decays like
    F(t) = Ae^(-b(t-t0)). 
    """
    y = np.piecewise(
        t, [t < t0, t >= t0], [lambda tp: 0.0, lambda tp: A * np.exp(-b * (tp - t0))]
    )
    return y



def Model_flare(Fres, t, model):

    """
    Returns the optimized flare according to a model.
    The model should also have the first parameter correspond to the peak of
    the flare; 
    """

    f = Model_exp

    if model == "exp":
        f = Model_exp
        # Initial guess roughly based on:
        #  - peak at maximum index
        #  - by flare's end, decay to 0.05 times the peak
        #  - start @ the beginning

        maxindex = np.argmax(Fres)

        initialguess = [
            np.max(Fres),
            -np.log(0.05) / (t[-FLARE_PAD] - t[maxindex]),
            t[maxindex],
        ]

    popt, pcov = curve_fit(f, t.astype(np.float), Fres, p0=initialguess)

    return (popt, np.array([f(time, *popt) for time in t]))


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


def E_flare(iamps, itimes, T_flare, LpS, LpF, R_star, model="none"):

    """
    The energy of the flare, following a model. 
    Takes in the amplitude,
    and pre-calculated values for the luminosity of the 
    star and the luminosity of the flare per unit area.
    """

    nans = np.isnan(iamps)
    amps = iamps[~nans]
    times = itimes[~nans]
    L_f = lambda amp: L_flare(T_flare, A_flare(amp, LpS, LpF, R_star))
    
    # Fmodel is a tuple; [0] is the optimized parameters,
    # [1] is the flare.
     

    if (model == "exp"):
        Fmodel = Model_flare(amps, times, "exp")
        impulsiveness = Fmodel[0][0]
        lumin = L_f(Fmodel[1])
        energy = simps(L_f(Fmodel[1]), times)  # not good enough, do explicit integral
    
    else: 
        impulsiveness = np.max(amps)
        energy = simps(L_f(amps), times)
    
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
    time = data[:, 0]
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
            amps = pdcflux_ratio[i - flare_dur - FLARE_PAD : i + FLARE_PAD]
            ts = time[i - flare_dur - FLARE_PAD : i + FLARE_PAD] * DAY
            (i_f, E_f) = E_flare(amps, ts, T_FLARE, LpS, LpF, R_STAR, "none")
            flare_energies.append(E_f / ERG)
            flare_impulses.append(i_f) # TODO check units!
            flare_times.append(time[i - flare_dur])
            in_flare = False
            flare_dur = 0

    # Print the flare energies to stdout

    for i in range(len(flare_energies)):
        print(
            "{:.6f}  {:.8e}  {:.8e}".format(flare_times[i], flare_energies[i], flare_impulses[i]))


if __name__ == "__main__":
    main()
