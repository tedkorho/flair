# DEPRECATED FUNCTIONS

def fit_exp(x, y):
    """
    Fits a simple exponential model to data
DEPRECATED
    """
    f = model_exp

    # Initial guess roughly based on:
    #  - peak at maximum index
    #  - by flare's end, decay to 0.05 times the peak
    #  - start @ the beginning

    maxindex = np.argmax(y)

    guess = [1.0, -np.log(0.05) / (x[-FLARE_PAD_1] - x[maxindex])]

    popt, pcov = curve_fit(f, x, y, p0=guess, method="trf")

    return popt


def fit_davenport(x, y):
    """
    Fits Davenport's model from Davenport et al. 2014 [TODO citation here]
    """

    f1 = model_double_exp
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
        fres_model = np.zeros(len(x))
        fres_model[mask0] = f0(x[mask0], *popt0)
        fres_model[x >= 0] = f1(x[x >= 0], *popt1)

    except RuntimeError as err:
        fres_model = fit_exp(x, y)
        print("{0}".format(err))

    return [*popt0, *popt1]


def Model_flare(fres, t, model="exp"):

    """
    Returns the optimized flare according to a model, and its energy.
    The model should also have the first parameter correspond to the peak of
    the flare. The model is fit using an interpolation of the flux residuals,
    by default normalized to the units used in Davenport (2014);
    some helper values are provided in the beginning.
    """
    
# TODO consider refactoring this s.t. it uses the same fn. as the multiple flare fitting

    interpol = interp1d(t, fres, fill_value="extrapolate")
    x = np.arange(t[0], t[-1], 10.0)
    y = interpol(x)
    maxx = x[np.argmax(y)]
    maxy = np.max(y)
    thalf = find_thalf(x, y, maxx, maxy)
    x = (x - maxx) / thalf
    y = y / maxy

    fres_model = np.zeros(len(x))
    ndeg = 1

    if model == "exp":
        f = model_exp
        ndeg = 2
        popt = fit_exp(x, y)

    if model == "Davenport":
        f = model_davenport
        ndeg = 2
        popt = fit_davenport(x,y)

    if model == "Davenport2":
        ndeg = 3
        f = lambda xp, a, b, c: a * Model_Davenport(b * (xp - c))
        popt = fit_davenport_2(x, y)
        
    fres_model = f(x, *popt) * maxy
    times = x * thalf + maxx
    xt = (t - maxx) / thalf
    fres_obs = f((t - maxx) / thalf, *popt)
    BIC = sqerr(fres, fres_obs*maxy)/(len(t) - 3) + 3.*np.log(len(t))

    if PLOT == True:
        plt.clf()
        plt.plot(x, fres_model)
        plt.plot(xt, fres, "ro")
        plt.legend(["Fit; BIC = {:.3f}".format(BIC), "Data"])
        plt.show(block=False)

    return (maxy, times, fres_model)


