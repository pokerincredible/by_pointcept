# python convert_femur_stl_to_s3dis.py \
#     --train_test_split \
#     --num_workers 12 \
#     --input_root /home/poker/workspace/11_third_projs/data/01_raw_cls \
#     --output_root /home/poker/workspace/11_third_projs/data/train_data/01_version/c5 \
#     --train_ratio 0.8 \
#     --minority_classes '2,3' \
#     --class_mode 5


# # 5 分类 + 默认少数类增强
# python convert_femur_stl_to_s3dis_v1.py \
#     --input_root /home/poker/workspace/11_third_projs/data/01_raw_cls \
#     --output_root /home/poker/workspace/11_third_projs/data/train_data/01_version_v2 \
#     --boundary_radius 0.005


python convert_v3.py \
    --input_root /home/poker/workspace/11_third_projs/data/01_raw_cls \
    --output_root /home/poker/workspace/11_third_projs/data/train_data/03_version \