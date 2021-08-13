import flare_energy as fe
import numpy as np
import argparse as ap
import matplotlib.pyplot as plt

def lcplot(t,f,color=""):
    '''
    Simple lightcurve plot function
    '''
    mp.rcParams['figure.figsize'] = [12, 6]
    plt.plot(t,f,color+".")
    plt.xlabel("Timestamp (days)")
    plt.ylabel(r"SAP flux ($e^-/s$)")

def lcminiplot(t,f,axs,i,j,color="",alpha=1.0):
    axs[i,j].plot(t,f,color+".")

def getargs():
    parser = ap.ArgumentParser()
    parser.add_argument("flarefile", metavar="ff", type=str)
    parser.add_argument("nflares", metavar="n", type=int)
    return parser.parse_args()

def main():
    args = getargs()
    data = np.loadtxt(args.flarefile)
    pdcflux = data[:, 1]
    time = data[:, 0]
    trend = data[:, 2]
    flarestamps = data[:, 3]
    pdcflux_ratio = (pdcflux - trend) / trend
    iflare = 1
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
            (i_f, E_f) = fe.E_flare(amps, ts, T_FLARE, LpS, LpF, R_STAR, model="exp")
            flare_energies.append(E_f / ERG)
            flare_impulses.append(i_f)
            flare_times.append(time[i - flare_dur])
            in_flare = False
            iflare += 1
            flare_dur = 0


        

    return

if __name__ == "__main__":
    main()
