flareout="out/temp.out"
energyout="out/energies_ekdra.out"
arrivalout="out/interarrival_times_ekdra.out"
flareout2="out/temp2.out"
flareout3="out/temp3.out"
lc_preamble="# LIGHTCURVE\n# timestamp | PDC flux | trend | is_flare"

for f in $(find input -name "*_lc.fits")
do
    echo "$basename $f"
    bn="$(basename -s .fits $f)"
	python3 src/candidates.py "$f" > "$flareout"
	python3 src/iteration.py "$flareout" > "$flareout2"
	echo "$lc_preamble" > "$flareout3"
    python3 src/iteration.py "$flareout2" --plot "out/$bn.png" >> "$flareout3"
	cp "$flareout3" "out/$bn.dat"
    rm "$flareout"
	rm "$flareout2"
	rm "$flareout3"
done

# ADD --show to the trend_iteration.py commands to view results during calculation
