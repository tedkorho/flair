energyout="out/energies.out"
arrivalout="out/interarrival_times.out"

echo "# Timestamp | Energy (erg) | Impulsiveness" > "$energyout"

for f in $(find out -name "*_lc.dat")
do
    echo "$f"
	python3 src/flare_energy.py "$f" "starinfo.fits" # >> "$energyout"
	# python3 src/flare_interarrival.py "$f" >> "$arrivalout"
done
