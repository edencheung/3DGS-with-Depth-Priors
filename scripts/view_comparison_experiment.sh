# Run experiment to compare training results of DNGaussian with multiple views
# Can disable regularisation in train_llff.py to do without regularisation too

for views in 3 6 9; do
   export NVIEWS="$views"
   for scene in fern flower fortress horns leaves orchids room trex; do
      bash scripts/run_llff.sh "data/llff/$scene" "output/llff/$scene/dpt_midas_$views" 0 || break 2
   done
done