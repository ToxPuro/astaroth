#!/bin/bash
src=$1
dst=$2
dims_x=$3
dims_y=$4

if [[ ! -e "$src" ]] || [[ ! -e "$dst" ]]; then
    printf "Bad arguments src:\"$src\" dst:\"$dst\""
    exit 1
fi

#Find the next free dirname
suffix=1
while true; do
    out_dir=$dst/batch.$suffix
    if [[ ! -x $out_dir ]]; then
        break
    fi
    suffix=$((suffix + 1))
done

mkdir -p $out_dir

find_newest_slice(){
    # This may be iffy if we have too many files in $src...
    ls -1t "$src"/*.slice | head -1
}

plot_script=../../analysis/viz_tools/plot

newest_slice=$(find_newest_slice)

echo "Moving files older than $newest_slice to $out_dir" >> render.log
find "$src" -not -newer "$newest_slice" -type f -exec mv -t "$out_dir" \{\} +
echo "Rendering frames" >> render.log
${plot_script} --output=$out_dir/frames --input $out_dir/* --dims $dims_x $dims_y
