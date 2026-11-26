# Run 3-view default DNGaussian to understand training dynamics

export EXPERIMENT_NAME="dpt_midas_train_inspection"
export NVIEWS="3"
for scene in fern flower fortress horns leaves orchids room trex; do
      bash scripts/run_llff.sh "data/llff/$scene" "output/llff/$scene/$EXPERIMENT_NAME" 0 || break 2

      for iteration in 1000 2000 3000 4000 5000; do
            python spiral.py -s "data/llff/$scene" \
            --model_path "output/llff/$scene/$EXPERIMENT_NAME" \
            -r 8 --near 10 --iteration "$iteration"

            python render.py -s "data/llff/$scene" \
            --model_path "output/llff/$scene/$EXPERIMENT_NAME" \
            -r 8 --near 10 --iteration "$iteration"
      done
done