# cd datasets || exit 0
# range_starts=0
# range_ends=(10 20 30)
# points=(5 10 15)
# for i in "${!range_ends[@]}"
# do
#     papermill create_data.ipynb /dev/null \
#         -p training_range_start ${range_starts} \
#         -p training_range_end ${range_ends[$i]} \
#         -p data_points ${points[$i]}
# done
# cd ..

# Trains only the baseline model - takes ~ 2 mins
# python3 train_networks.py --opt baseline
cd baseline_mlp_cnn || exit 0
jupyter nbconvert --to notebook --execute test_MLP.ipynb \
    --inplace
jupyter nbconvert --to notebook --execute test_CNN.ipynb \
    --inplace
cd ..

# Uncomment this if you want to train all networks
# Takes long time
# python3 train_networks.py --opt all