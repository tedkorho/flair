energyout="out/energies.out"
arrivalout="out/interarrival_times.out"

for f in $(find out -name "*_lc.dat")
do
    echo "$f"
    echo "# Timestamp | Energy (erg)" > "$energyout"
	python3 src/flare_energy.py "$f" "starinfo.fits" >> "$energyout"
	# python3 src/flare_interarrival.py "$f" >> "$arrivalout"
done
