for f in $(find input -name "*_lc.fits")
do
	echo "$f"
	python3 src/starinfo.py "$f"
	break
done
