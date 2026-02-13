$ErrorActionPreference = "Stop"

param(
  [string]$DataRoot = "D:\SoccerNet\jersey-2023",
  [string]$TinyOut = ".\data\tiny5"
)

Write-Host "Creating tiny subset..."
python scripts/make_tiny_subset.py --data_root $DataRoot --split train --num_clips 5 --out_dir $TinyOut

Write-Host "Running sanity training loop..."
python src/train_sanity.py --config configs/sanity.yaml
