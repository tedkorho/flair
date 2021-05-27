from astropy.io import fits
import argparse as ap

parser = ap.ArgumentParser(description="View some stellar stats from a TESS FITS file")
parser.add_argument("inputfile", metavar="if", type=str, help="Input file")
args = parser.parse_args()

hdul = fits.open(args.inputfile)
hdr0 = hdul[0].header
T_STAR = hdr0['TEFF']
R_STAR = hdr0['RADIUS']
OBJECT = hdr0['OBJECT']

hdul.close()

print("TESS Object ID  :".format())
print("Star temperature:\t{}".format(T_STAR))
print("Star radius     :\t{}".format(R_STAR))
