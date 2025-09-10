# python model/TC_predict/inference.py --model_type pc_with_no_rgb --data_dir data --output_dir output/pc_with_no_rgb --sample_ratio 0.2
# python model/TC_predict/inference.py \
#     --model_type origin \
#     --data_dir data \
#     --output_dir output/image_occlusion \
#     --sample_ratio 0.2 \
#     --roboustness_test \
#     --roboustness_test_mode image_occlusion

# python model/TC_predict/inference.py \
#     --model_type origin \
#     --data_dir data \
#     --output_dir output/pointcloud_sparse \
#     --sample_ratio 0.2 \
#     --roboustness_test \
#     --roboustness_test_mode pointcloud_sparse

# python model/TC_predict/inference.py \
#     --model_type origin \
#     --data_dir data \
#     --output_dir output/pointcloud_noise \
#     --sample_ratio 0.2 \
#     --roboustness_test \
#     --roboustness_test_mode pointcloud_noise

python model/TC_predict/inference.py \
    --model_type no_modulation \
    --data_dir data \
    --output_dir output/no_modulation \
    --sample_ratio 0.2 \
