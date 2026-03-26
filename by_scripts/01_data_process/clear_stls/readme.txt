从原始的 stl 里面选取最大的联通区域
conda activate pt209
python by_scripts/01_data_process/clear_stls/keep_largest_component.py `
  --root "data/01_raw_cls" `
  --out_root "data/02_raw_cls_clear/version0" `
  --only_name femur.stl `
  --vertices_match_mode surface `
  --vertices_dist_thresh 1.0 `
  --dry_run 0


将清理后的模型转为5分类的数据，并保留三视图
python by_scripts/01_data_process/clear_stls/convert_cleaned_to_s3dis_v3.py `
  --input_root "data/02_raw_cls_clear/version0" `
  --output_root "data/03_train_data/version0" `
  --viz --vertices_nan_policy zero