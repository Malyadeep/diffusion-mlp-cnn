cd datasets || exit 0
rm -rf training_size* figures
cd ..

for case in "baseline_mlp_cnn" "increased_training_examples" "mlp_depth_width"
do
    cd ${case} || exit 0
    rm -rf trained_models training_performance figures
    cd ..
done