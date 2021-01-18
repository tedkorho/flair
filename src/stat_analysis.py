import numpy as np
import matplotlib.pyplot as plt

energies = np.loadtxt("energies_ekdra.out")[:,0]
interarrivals = np.loadtxt("interarrival_times_ekdra.out")

logenergies = np.log10(energies)
(n, logbins, tmp) = plt.hist(logenergies)
plt.show()
logn = np.log10(n)
coeffs = np.polyfit(logbins[:-1], logn, 1)
fit = np.poly1d(coeffs)
brange = np.arange(min(logbins),max(logbins))
plt.plot(logbins[:-1], logn, "ro")
plt.plot(brange, fit(brange))
plt.text(34,0.0,"Power-law coefficient {:.1f}".format(coeffs[1]))
plt.xlabel("log(E)")
plt.ylabel(r"$\log_{10}(n)$")
plt.show()

(n, bins, tmp)= plt.hist(interarrivals)
plt.show()
