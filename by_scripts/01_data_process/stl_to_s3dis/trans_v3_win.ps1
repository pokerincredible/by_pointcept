# Windows PowerShell one-click runner for dataset conversion
#
# 默认：清理后目录（femur_largest.stl + vertices_largest.txt）-> npy + triview
#   使用 clear_stls/convert_cleaned_to_s3dis_v3.py（坐标与标签来自 vertices_largest，不做 STL 近邻匹配）
#
# 原始：data/01_raw_cls 风格（femur.stl + vertices.txt）-> 使用本目录下的 convert_v3.py

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$dataProcessDir = Split-Path $scriptDir -Parent

# "cleaned" = 02_raw_cls_clear/... ； "raw" = 01_raw_cls 等
$Pipeline = "cleaned"

# Adjust these paths if your dataset/output location differs
$InputRoot = "D:\Projects\by_pointcept\data\02_raw_cls_clear\version1"
$OutputRoot = "D:\Projects\by_pointcept\data\03_train_data\version1"

# Visualization can be slow for large datasets; set to $false to disable
$Viz = $true
$VizMaxPoints = 300000
$VizPointSize = 0.2
$VizBoundary34 = $false
$VizBoundary34Radius = 0.005
$VizBoundary34MaxPointsDet = 200000
$VizBoundary34PointSize = 0.3

Write-Host "Pipeline   : $Pipeline"
Write-Host "InputRoot  : $InputRoot"
Write-Host "OutputRoot : $OutputRoot"
Write-Host "Viz        : $Viz"

if ($Pipeline -eq "cleaned") {
  $pyFile = Join-Path $dataProcessDir "clear_stls\convert_cleaned_to_s3dis_v3.py"
  if (-not (Test-Path -LiteralPath $pyFile)) {
    throw "Script not found: $pyFile"
  }

  $pyArgs = @(
    "`"$pyFile`"",
    "--input_root", "`"$InputRoot`"",
    "--output_root", "`"$OutputRoot`"",
    "--normal_mode", "estimate",
    "--normal_knn", "30",
    "--vertices_nan_policy", "zero"
  )

  if ($Viz) {
    $pyArgs += @(
      "--viz",
      "--viz_max_points", "$VizMaxPoints",
      "--viz_point_size", "$VizPointSize"
    )
  }

  if ($Viz -and $VizBoundary34) {
    $pyArgs += @(
      "--viz_boundary34",
      "--viz_boundary34_radius", "$VizBoundary34Radius",
      "--viz_boundary34_max_points_det", "$VizBoundary34MaxPointsDet",
      "--viz_boundary34_point_size", "$VizBoundary34PointSize"
    )
  }
}
else {
  # raw: femur.stl + vertices.txt
  $pyFile = Join-Path $scriptDir "convert_v3.py"
  $pyArgs = @(
    "`"$pyFile`"",
    "--input_root", "`"$InputRoot`"",
    "--output_root", "`"$OutputRoot`""
  )

  if ($Viz) {
    $pyArgs += @(
      "--viz",
      "--viz_max_points", "$VizMaxPoints",
      "--viz_point_size", "$VizPointSize"
    )
  }

  if ($Viz -and $VizBoundary34) {
    $pyArgs += @(
      "--viz_boundary34",
      "--viz_boundary34_radius", "$VizBoundary34Radius",
      "--viz_boundary34_max_points_det", "$VizBoundary34MaxPointsDet",
      "--viz_boundary34_point_size", "$VizBoundary34PointSize"
    )
  }
}

python @pyArgs
