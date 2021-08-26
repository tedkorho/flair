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


def Model_exp(t, A, b):
    """
    Models a flare that starts from t0, has a peak A, and decays like
    F(t) = Ae^(-b(t-t0)). 
    """
    y = np.piecewise(
        t, [t < 0, t >= 0], [lambda tp: 0.0, lambda tp: A * np.exp(-b * tp)]
    )
    return y

def Model_Davenport(t, A1, A2, k1, k2, a1, a2, a3, a4):
    """
    Models a flare after the model in Davenport et al., 2014:
    a 4th degree polynomial and a 2-regime decay afterwards.
    Assumes the flare peaks at t=0.
    """

    return np.piecewise(t, [t < 0., t>= 0.],
            [lambda tp: a1*tp + a2*tp**2 + a3*tp**3 + a4*tp**4 + (A1+A2),
            lambda tp: A1*np.exp(-k1*tp) + A2*np.exp(-k2*tp)
            ]
    )

def Model_Davenport_adapted(t, A1, A2, k1, k2):
    """
    Similar to Davenport, but with a simpler linear rise period (to avoid ill conditioned fits)
    """

    return np.piecewise(t, [t < 0, t>= 0],
        [lambda tp: 0.1,
        lambda tp: A1*np.exp(-k1*tp) + A2*np.exp(-k2*tp)
        ]
    )


def Model_flare(Fres, t, model="exp"):

    """
    Returns the optimized flare according to a model, and its energy.
    The model should also have the first parameter correspond to the peak of
    the flare. The model is fit using an interpolation of the flux residuals,
    by default normalized to the units used in Davenport (2014);
    some helper values are provided in the beginning.
    """
    
    interpol = interp1d(t, Fres, fill_value="extrapolate")
    x = np.arange(t[0], t[-1], 1.0/(24.*60.))
    y = interpol(x)
    maxx = x[np.argmax(y)]
    maxy = np.max(y)
    
    # Find t_1/2 and use it for normalization

    ctr = 0
    t1 = x[0]
    t2 = maxx
    for i in range(len(x)):
        if ctr == 0 and y[i] > 0.5*maxy:
            ctr = 1
            t1 = x[i]
        if ctr == 1 and y[i] < 0.5*maxy:
            t2 = x[i]
            break

    thalf = t2 - t1
    x = (x-maxx)/thalf
    y = y/maxy

    if model == "exp":
        
        f = Model_exp
        
        # Initial guess roughly based on:
        #  - peak at maximum index
        #  - by flare's end, decay to 0.05 times the peak
        #  - start @ the beginning

        maxindex = np.argmax(y)

        initialguess = [
            1.,
            -np.log(0.05) / (x[-FLARE_PAD] - x[maxindex])
        ]
    
    if model == "Davenport":
        f = Model_Davenport #(t, A1, A2, k1, k2, a1, a2, a3, a4)
        
        # Initial guess: sharper exp higher, maximum at t0 
        
        maxindex = np.argmax(y)

        initialguess = [
            3./4., 
            1./4., 
            -np.log(0.05) / (x[-FLARE_PAD] - x[maxindex]),
            -np.log(0.1) / (x[-FLARE_PAD] - x[maxindex]),
            1., 1., 1., 1.,
        ]

    if model == "Davenport_adapted":
        f = Model_Davenport_adapted
        
        # Initial guess: exponential terms equal, maximum at t0
        
        maxindex = np.argmax(Fres)
        initialguess = [
            np.max(Fres)/2., 
            np.max(Fres)/2., 
            -np.log(0.02) / (t[-FLARE_PAD] - t[maxindex]),
            -np.log(0.1) / (t[-FLARE_PAD] - t[maxindex]),
            t[maxindex]
        ]


    popt, pcov = curve_fit(f, x, y, p0=initialguess, method='trf')

    Fres_model = f(x, *popt)*maxy
    time = x*thalf + maxx
    
    plt.plot(time, Fres_model)
    plt.plot(t, Fres)
    plt.show()

    return (maxy, time, Fres_model)


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

    # Fmodel is a tuple; [0] is the maximum,
    # [1] is the interpolated times,
    # [2] is the modelled flare as an array.
    
#TODO make this compatible

    if model == "exp":
        Fmodel = Model_flare(amps, times, "exp")
        impulsiveness = Fmodel[0]/2
        lumin = L_f(Fmodel[2])
        energy = simps(lumin, Fmodel[1])

    elif model == "Davenport":
        Fmodel = Model_flare(amps, times, "Davenport")
        impulsiveness = Fmodel[0]/2
        lumin = L_f(Fmodel[2])
        energy = simps(lumin, Fmodel[1]) # same here!

    elif model == "Davenport_adapted":
        Fmodel = Model_flare(amps, times, "Davenport_adapted")
        impulsiveness = Fmodel[0]/2
        lumin = L_f(Fmodel[2])
        energy = simps(lumin, Fmodel[1]) # same here!

    else:
        impulsiveness = np.max(amps)/2
        interp = interp1d(times, amps)
        x = np.linspace(np.min(times), np.max(times), 0.1/(24.*60.))
        y = interp(x)
        energy = simps(L_f(y), x)

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
            (i_f, E_f) = E_flare(amps, ts, T_FLARE, LpS, LpF, R_STAR, model="Davenport")
            flare_energies.append(E_f / ERG)
            flare_impulses.append(i_f)
            flare_times.append(time[i - flare_dur])
            in_flare = False
            flare_dur = 0

    # Print the flare energies to stdout

    for i in range(len(flare_energies)):
        print(
            "{:.6f}  {:.8e}  {:.8e}".format(
                flare_times[i], flare_energies[i], flare_impulses[i]
            )
        )


if __name__ == "__main__":
    main()
