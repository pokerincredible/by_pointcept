从原始的 stl 里面选取最大的联通区域
class 0 → 红色（[0.894, 0.102, 0.110]）
class 1 → 蓝色（[0.216, 0.494, 0.722]）
class 2 → 绿色（[0.302, 0.686, 0.290]）
class 3 → 紫色（[0.596, 0.306, 0.639]）
class 4 → 橙色（[1.000, 0.498, 0.000]）
conda activate pt209
python by_scripts/01_data_process/clear_stls/keep_largest_component.py `
  --root "data/01_raw_cls" `
  --out_root "data/02_raw_cls_clear/version0" `
  --only_name femur.stl `
  --vertices_match_mode surface `
  --vertices_dist_thresh 1.0 `
  --dry_run 0

python by_scripts/01_data_process/clear_stls/align_stl.py `
  --input_root "D:\Projects\by_pointcept\data\02_raw_cls_clear\version1" `
  --out_root "D:\Projects\by_pointcept\data\02_raw_cls_clear_aligned\version1" `
  --align_z 1 --align_mode level02 `
  --semantic_class_a 0 --semantic_class_b 2 `
  --level_tip_ratio 0.12 `
  --auto_mirror_zy_rule 1 `
  --auto_flip_lr 1 `
  --force_mirror_zy_cases "1934031_CHEN_ZHUO,2153208_HAO_LIN_YU,1762897_GUO_SONG,1796572_WANG_HONG_KUN,1990863_QI_SHU_TING,2160549_MAO_XIAO_FENG,2182010_TANG_ZI_HAN,2374125_REN_YING,2445088_WANG_BEN_SHUANG,2578743_ZHANG_YOU,2587389_ZHOU_XIAO_QI,2588986_LU_YAN_BIN,2594470_LIN_MENG_YANG,2620893_WU_XIU_NA,2630525_LU_SHUAI,2633604_ZHU_BO_HAN,2674846_LV_XIN,2696355_LI_ZE_YU,2726979_YAN_ZHI_CHUN,2756115_YANG_BO,2784455_HUANG_CHUN_JIANG,2798933_LIU_SHEN_YING,2844997_LIU_LU,2845047_GAO_YONG,2873001_LI_YONG_YE,2910938_WANG_XING_YE,2991808_LI_XIAO,2995309_MA_HAO,3043008_XU_HONG_QIANG,3090752_SONG_LI_NA,3190555_YANG_SHU_XIANG,3198056_YU_CAI_HONG,3253100_HE_YU_QING,3263027_XU_HONG_XIA,3270055_YANG_TING,3280090_WAN_XIAO_HUI,3295606_WANG_LEI,3295834_TAN_ZHEN_TAO,3299008_LI_YAN_XIN,3302061_JIANG_CHANG_MIN,3345240_WANG_LI_KUN,3347153_SHI_JI_YANG,3356849_YI_JIAN_WEI,3361199_ZHAO_SHENG,3369582_YANG_CHUN_TAO" `
  --smooth_mesh 1 --smooth_method taubin --smooth_iters 10


python by_scripts/01_data_process/clear_stls/align_stl.py `
  --input_root "D:\Projects\by_pointcept\data\02_raw_cls_clear\version1" `
  --out_root "D:\Projects\by_pointcept\data\02_raw_cls_clear_aligned\version1" `
  --align_z 1 --align_mode level02 `
  --semantic_class_a 0 --semantic_class_b 2 `
  --level_tip_ratio 0.12 `
  --auto_mirror_zy_rule 0 `
  --smooth_mesh 1 --smooth_method taubin --smooth_iters 10 `
  --project_vertices_to_mesh 0

python by_scripts/01_data_process/clear_stls/align_stl.py `
  --input_root "D:\Projects\by_pointcept\data\02_raw_cls_clear\version1" `
  --out_root "D:\Projects\by_pointcept\data\02_raw_cls_clear_aligned\version2" `
  --align_z 1 --align_mode level02 `
  --semantic_class_a 0 --semantic_class_b 2 `
  --level_tip_ratio 0.12 `
  --auto_mirror_zy_rule 0 `
  --smooth_mesh 1 --smooth_method taubin --smooth_iters 10

python by_scripts/01_data_process/clear_stls/align_stl.py `
  --input_root "D:\Projects\by_pointcept\data\02_raw_cls_clear\version1" `
  --out_root "D:\Projects\by_pointcept\data\02_raw_cls_clear_aligned\version3" `
  --align_z 1 --align_mode level02 `
  --semantic_class_a 0 --semantic_class_b 2 `
  --level_tip_ratio 0.12 `
  --auto_mirror_zy_rule 0 `
  --extra_rot180_cases_file "D:\Projects\by_pointcept\data\rot.txt" `
  --extra_flip_zy_after_rot180 1 `
  --smooth_mesh 1 --smooth_method taubin --smooth_iters 10


python by_scripts/01_data_process/clear_stls/align_stl.py `
  --input_root "D:\Projects\by_pointcept\data\02_raw_cls_clear\version1" `
  --out_root "D:\Projects\by_pointcept\data\02_raw_cls_clear_aligned\version4" `
  --align_z 1 --align_mode level02 `
  --semantic_class_a 0 --semantic_class_b 2 `
  --level_tip_ratio 0.12 `
  --auto_mirror_zy_rule 0 `
  --extra_rot180_cases_file "D:\Projects\by_pointcept\data\rot.txt" `
  --extra_flip_zy_after_rot180 1 `
  --smooth_mesh 1 --smooth_method taubin --smooth_iters 10 `
  --smooth_strength_file "D:\Projects\by_pointcept\data\smooth.txt" `

python by_scripts/01_data_process/clear_stls/cp_img.py


python by_scripts/01_data_process/clear_stls/convert_cleaned_to_s3dis_v3.py `
  --input_root "D:\Projects\by_pointcept\data\02_raw_cls_clear_aligned\version4" `
  --output_root "D:\Projects\by_pointcept\data\03_train_data_aligned\version4" `
  --viz --vertices_nan_policy zero


将清理后的模型转为5分类的数据，并保留三视图
python by_scripts/01_data_process/clear_stls/convert_cleaned_to_s3dis_v3.py `
  --input_root "D:\Projects\by_pointcept\data\02_raw_cls_clear_aligned\version1" `
  --output_root "D:\Projects\by_pointcept\data\03_train_data_aligned\version1" `
  --viz --vertices_nan_policy zero