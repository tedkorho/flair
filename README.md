#FLARE DETECTION TOOL

Teo Korhonen, 2020-2021.

Detects flares from a collection of lightcurves stored in some directory structure, and estimates their energies based on the star's luminosity and a black-body model of the flare radiation in the TESS band.

## INSTRUCTIONS

1) Store .fits files containing the lightcurves in any directory structure in. These need to follow the TESS Science Data Products Description Document specifications\*.

2) Edit the `src/params.dat` file to match the specifications of the star and/or the flare detection.

3) Detect flare candidates in each lightcurve by running the `detect.sh` script in the root folder. By default, this also produces plots of each lightcurve. The results for each lightcurve are stored in human-readable (and editable) format in the `out` directory.

4) If there are issues, such as flare candidates splitting up to several pieces due to stray NaN values in the dataset, you may repair them manually here. (This happened in e.g. certain EK Draconis lightcurves)

5) Calculate the energies of these flare candidates by running the `energy.sh` script. This gathers all the flares in one output file.

## POSSIBLE CAUSES OF ERROR

- Make sure the base filenames of each lightcurve are distinct - otherwise this may cause collisions in the output directory.

- As stated, stray NaNs can cause errors.

- Rarely, for reasons beyond my humble mind, .fits files may break after this program opens them, causing repeat runs of the detection script to fail. Keep a backup on the ready.

## FURTHER MODIFICATIONS

If the script is modified for some other satellite mission in the future, make sure to find its response curve somewhere - the energy estimation relies on this.

Contact teo.korhonen@gmail.com if you have anything to ask.



\* See: a document of the same name in https://heasarc.gsfc.nasa.gov/docs/tess/documentation.html
