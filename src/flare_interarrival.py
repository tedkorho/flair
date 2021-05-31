import numpy as np
import argparse as ap


def interflare_time(flux, flare_points, time):

    # Returns a series of interarrival times between flares in days

    cadence_time = time[1] - time[0]
    if_times = []
    in_flare = False
    ctr = 0

    for i in range(len(time)):
        if flare_points[i] == 1 and ctr > 1:
            in_flare = True
            if_times.append(cadence_time * ctr)
            ctr = 0
        if flare_points[i] == 1:
            in_flare = True
            ctr = 0
        elif flare_points[i] == 0 and in_flare:
            in_flare = False
            ctr += 1
        elif np.isnan(flux[i]):
            ctr = 0
        else:
            ctr += 1

    return if_times


parser = ap.ArgumentParser(description="Gather flare interarrival times to stdout")
parser.add_argument("inputfile", metavar="if", type=str, help="Input file")
args = parser.parse_args()

data = np.loadtxt(args.inputfile)
flux = data[:, 1]
time = data[:, 0]
trend = data[:, 2]
flarestamps = data[:, 3]

interarrival_times = interflare_time(flux, flarestamps, time)
for i in range(len(interarrival_times)):
    print("{:.5f}".format(interarrival_times[i]))
