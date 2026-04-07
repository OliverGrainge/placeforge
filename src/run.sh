source ../.venv/bin/activate

python -m cli datapipeline sf_xl_gsvcities_cossim_0p10_train
python -m cli datapipeline sf_xl_gsvcities_cossim_0p30_train
python -m cli datapipeline sf_xl_gsvcities_cossim_0p50_train


python -m cli train /home/oliver/github/placeforge/src/configs/train/experiments/sf_xl_gsvcities_0p10_train.yaml
python -m cli train /home/oliver/github/placeforge/src/configs/train/experiments/sf_xl_gsvcities_0p30_train.yaml
python -m cli train /home/oliver/github/placeforge/src/configs/train/experiments/sf_xl_gsvcities_0p50_train.yaml
