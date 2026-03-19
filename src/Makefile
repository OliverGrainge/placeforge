
SHELL := /bin/bash

# sf_xl_datapipeline disabled entries:
#   sf_xl_cosplace, sf_xl_eigenplaces, sf_xl_intra_inter,
#   sf_xl_intra_inter_cell15_cos03_sg64, sf_xl_intra_inter_cell20_cos03_sg64,
#   sf_xl_intra_inter_cell15_cos04_sg64, sf_xl_intra_inter_cell15_cos05_sg64,
#   sf_xl_intra_inter_cell15_cos06_sg64, sf_xl_intra_inter_cell15_cos03_sg16,
#   sf_xl_intra_inter_cell15_cos03_sg32, sf_xl_intra_inter_cell15_cos03_sg128,
#   sf_xl_intra_inter_cell15_cos03_sg512, sf_xl_intra_inter_cell10_cos03_sg64
sf_xl_datapipeline:
	@source ../.venv/bin/activate && \
	for pipeline in \
		sf_xl_intra_inter_cell10_cos02_sg16 \
		sf_xl_intra_inter_cell10_cos02_sg32 \
		sf_xl_intra_inter_cell10_cos02_sg64 \
	; do \
		echo "--- $$pipeline ---" && python -m cli datapipeline $$pipeline; \
	done

sf_xl_small_datapipeline:
	source ../.venv/bin/activate && \
	echo "--- sf_xl_small_intra_inter_cell10_cos03_sg64 ---" && \
	python -m cli datapipeline sf_xl_small_intra_inter && \
	echo "--- sf_xl_small_intra_inter_cell15_cos03_sg64 (baseline) ---" && \
	python -m cli datapipeline sf_xl_small_intra_inter_cell15_cos03_sg64 && \
	echo "--- sf_xl_small_intra_inter_cell20_cos03_sg64 ---" && \
	python -m cli datapipeline sf_xl_small_intra_inter_cell20_cos03_sg64 && \
	echo "--- sf_xl_small_intra_inter_cell15_cos04_sg64 ---" && \
	python -m cli datapipeline sf_xl_small_intra_inter_cell15_cos04_sg64 && \
	echo "--- sf_xl_small_intra_inter_cell15_cos05_sg64 ---" && \
	python -m cli datapipeline sf_xl_small_intra_inter_cell15_cos05_sg64 && \
	echo "--- sf_xl_small_intra_inter_cell15_cos06_sg64 ---" && \
	python -m cli datapipeline sf_xl_small_intra_inter_cell15_cos06_sg64 && \
	echo "--- sf_xl_small_intra_inter_cell15_cos03_sg16 ---" && \
	python -m cli datapipeline sf_xl_small_intra_inter_cell15_cos03_sg16 && \
	echo "--- sf_xl_small_intra_inter_cell15_cos03_sg32 ---" && \
	python -m cli datapipeline sf_xl_small_intra_inter_cell15_cos03_sg32 && \
	echo "--- sf_xl_small_intra_inter_cell15_cos03_sg128 ---" && \
	python -m cli datapipeline sf_xl_small_intra_inter_cell15_cos03_sg128 && \
	echo "--- sf_xl_small_intra_inter_cell15_cos03_sg512 ---" && \
	python -m cli datapipeline sf_xl_small_intra_inter_cell15_cos03_sg512



sf_xl_small_analyse:
	source ../.venv/bin/activate && \
	python -m cli analyse sf_xl_small_cosplace && \
	python -m cli analyse sf_xl_small_eigenplaces && \
	python -m cli analyse sf_xl_small_intra_inter --image-embedding-name sf_xl_small --place-embedding-name sf_xl_intra_inter && \
	python -m cli analyse sf_xl_small_intra_inter_cell10_cos03_sg64 --image-embedding-name sf_xl_small --place-embedding-name sf_xl_small_intra_inter_cell10_cos03_sg64 && \
	python -m cli analyse sf_xl_small_intra_inter_cell15_cos03_sg64 --image-embedding-name sf_xl_small --place-embedding-name sf_xl_small_intra_inter_cell15_cos03_sg64 && \
	python -m cli analyse sf_xl_small_intra_inter_cell20_cos03_sg64 --image-embedding-name sf_xl_small --place-embedding-name sf_xl_small_intra_inter_cell20_cos03_sg64 && \
	python -m cli analyse sf_xl_small_intra_inter_cell15_cos04_sg64 --image-embedding-name sf_xl_small --place-embedding-name sf_xl_small_intra_inter_cell15_cos04_sg64 && \
	python -m cli analyse sf_xl_small_intra_inter_cell15_cos05_sg64 --image-embedding-name sf_xl_small --place-embedding-name sf_xl_small_intra_inter_cell15_cos05_sg64 && \
	python -m cli analyse sf_xl_small_intra_inter_cell15_cos06_sg64 --image-embedding-name sf_xl_small --place-embedding-name sf_xl_small_intra_inter_cell15_cos06_sg64 && \
	python -m cli analyse sf_xl_small_intra_inter_cell15_cos03_sg16 --image-embedding-name sf_xl_small --place-embedding-name sf_xl_small_intra_inter_cell15_cos03_sg16 && \
	python -m cli analyse sf_xl_small_intra_inter_cell15_cos03_sg32 --image-embedding-name sf_xl_small --place-embedding-name sf_xl_small_intra_inter_cell15_cos03_sg32 && \
	python -m cli analyse sf_xl_small_intra_inter_cell15_cos03_sg128 --image-embedding-name sf_xl_small --place-embedding-name sf_xl_small_intra_inter_cell15_cos03_sg128 && \
	python -m cli analyse sf_xl_small_intra_inter_cell15_cos03_sg512 --image-embedding-name sf_xl_small --place-embedding-name sf_xl_small_intra_inter_cell15_cos03_sg512

sf_xl_small_compare:
	source ../.venv/bin/activate && \
	python -m cli compare \
		sf_xl_small_cosplace \
		sf_xl_small_eigenplaces \
		sf_xl_small_intra_inter \
		sf_xl_small_intra_inter_cell10_cos03_sg64 \
		sf_xl_small_intra_inter_cell15_cos03_sg64 \
		sf_xl_small_intra_inter_cell20_cos03_sg64 \
		sf_xl_small_intra_inter_cell15_cos04_sg64 \
		sf_xl_small_intra_inter_cell15_cos05_sg64 \
		sf_xl_small_intra_inter_cell15_cos06_sg64 \
		sf_xl_small_intra_inter_cell15_cos03_sg16 \
		sf_xl_small_intra_inter_cell15_cos03_sg32 \
		sf_xl_small_intra_inter_cell15_cos03_sg128 \
		sf_xl_small_intra_inter_cell15_cos03_sg512

sf_xl_analyse:
	source ../.venv/bin/activate && \
	python -m cli analyse sf_xl_cosplace --image-embedding-name sf_xl --place-embedding-name sf_xl_cosplace && \
	python -m cli analyse sf_xl_eigenplaces --image-embedding-name sf_xl --place-embedding-name sf_xl_eigenplaces && \
	python -m cli analyse sf_xl_intra_inter --image-embedding-name sf_xl --place-embedding-name sf_xl_intra_inter && \
	python -m cli analyse sf_xl_intra_inter_cell10_cos03_sg64 --image-embedding-name sf_xl --place-embedding-name sf_xl_intra_inter_cell10_cos03_sg64 && \
	python -m cli analyse sf_xl_intra_inter_cell15_cos03_sg64 --image-embedding-name sf_xl --place-embedding-name sf_xl_intra_inter_cell15_cos03_sg64 && \
	python -m cli analyse sf_xl_intra_inter_cell20_cos03_sg64 --image-embedding-name sf_xl --place-embedding-name sf_xl_intra_inter_cell20_cos03_sg64 && \
	python -m cli analyse sf_xl_intra_inter_cell15_cos04_sg64 --image-embedding-name sf_xl --place-embedding-name sf_xl_intra_inter_cell15_cos04_sg64 && \
	python -m cli analyse sf_xl_intra_inter_cell15_cos05_sg64 --image-embedding-name sf_xl --place-embedding-name sf_xl_intra_inter_cell15_cos05_sg64 && \
	python -m cli analyse sf_xl_intra_inter_cell15_cos06_sg64 --image-embedding-name sf_xl --place-embedding-name sf_xl_intra_inter_cell15_cos06_sg64 && \
	python -m cli analyse sf_xl_intra_inter_cell15_cos03_sg16 --image-embedding-name sf_xl --place-embedding-name sf_xl_intra_inter_cell15_cos03_sg16 && \
	python -m cli analyse sf_xl_intra_inter_cell15_cos03_sg32 --image-embedding-name sf_xl --place-embedding-name sf_xl_intra_inter_cell15_cos03_sg32 && \
	python -m cli analyse sf_xl_intra_inter_cell15_cos03_sg128 --image-embedding-name sf_xl --place-embedding-name sf_xl_intra_inter_cell15_cos03_sg128 && \
	python -m cli analyse sf_xl_intra_inter_cell15_cos03_sg512 --image-embedding-name sf_xl --place-embedding-name sf_xl_intra_inter_cell15_cos03_sg512 && \
	python -m cli analyse sf_xl_intra_inter_cell10_cos02_sg16 --image-embedding-name sf_xl --place-embedding-name sf_xl_intra_inter_cell10_cos02_sg16 && \
	python -m cli analyse sf_xl_intra_inter_cell10_cos02_sg32 --image-embedding-name sf_xl --place-embedding-name sf_xl_intra_inter_cell10_cos02_sg32 && \
	python -m cli analyse sf_xl_intra_inter_cell10_cos02_sg64 --image-embedding-name sf_xl --place-embedding-name sf_xl_intra_inter_cell10_cos02_sg64

sf_xl_compare:
	source ../.venv/bin/activate && \
	python -m cli compare \
		sf_xl_cosplace \
		sf_xl_eigenplaces \
		sf_xl_intra_inter \
		sf_xl_intra_inter_cell10_cos03_sg64 \
		sf_xl_intra_inter_cell15_cos03_sg64 \
		sf_xl_intra_inter_cell20_cos03_sg64 \
		sf_xl_intra_inter_cell15_cos04_sg64 \
		sf_xl_intra_inter_cell15_cos05_sg64 \
		sf_xl_intra_inter_cell15_cos06_sg64 \
		sf_xl_intra_inter_cell15_cos03_sg16 \
		sf_xl_intra_inter_cell15_cos03_sg32 \
		sf_xl_intra_inter_cell15_cos03_sg128 \
		sf_xl_intra_inter_cell15_cos03_sg512 \
		sf_xl_intra_inter_cell10_cos02_sg16 \
		sf_xl_intra_inter_cell10_cos02_sg32 \
		sf_xl_intra_inter_cell10_cos02_sg64

SRC_DIR := /home/oliver/github/placeforge/src

sf_xl_small_train:
	@source ../.venv/bin/activate && \
	for cfg in \
		$(SRC_DIR)/configs/sf_xl_small/cosplace.yaml \
		$(SRC_DIR)/configs/sf_xl_small/eigenplaces.yaml \
		$(SRC_DIR)/configs/sf_xl_small/intra_inter_cell10_cos03_sg64.yaml \
		$(SRC_DIR)/configs/sf_xl_small/intra_inter_cell15_cos03_sg64.yaml \
		$(SRC_DIR)/configs/sf_xl_small/intra_inter_cell20_cos03_sg64.yaml \
		$(SRC_DIR)/configs/sf_xl_small/intra_inter_cell15_cos04_sg64.yaml \
		$(SRC_DIR)/configs/sf_xl_small/intra_inter_cell15_cos05_sg64.yaml \
		$(SRC_DIR)/configs/sf_xl_small/intra_inter_cell15_cos06_sg64.yaml \
		$(SRC_DIR)/configs/sf_xl_small/intra_inter_cell15_cos03_sg16.yaml \
		$(SRC_DIR)/configs/sf_xl_small/intra_inter_cell15_cos03_sg32.yaml \
		$(SRC_DIR)/configs/sf_xl_small/intra_inter_cell15_cos03_sg128.yaml \
		$(SRC_DIR)/configs/sf_xl_small/intra_inter_cell15_cos03_sg512.yaml \
	; do \
		while [ $$(jobs -r | wc -l) -ge 4 ]; do sleep 10; done; \
		echo "--- Starting $$cfg ---" && python -m cli train $$cfg & \
	done; \
	wait && echo "All training runs complete"

# sf_xl_train disabled entries:
#   cosplace.yaml, eigenplaces.yaml, intra_inter_cell10_cos03_sg64.yaml,
#   intra_inter_cell15_cos03_sg64.yaml, intra_inter_cell20_cos03_sg64.yaml,
#   intra_inter_cell15_cos04_sg64.yaml, intra_inter_cell15_cos05_sg64.yaml,
#   intra_inter_cell15_cos06_sg64.yaml, intra_inter_cell15_cos03_sg16.yaml,
#   intra_inter_cell15_cos03_sg32.yaml, intra_inter_cell15_cos03_sg128.yaml,
#   intra_inter_cell15_cos03_sg512.yaml
sf_xl_train:
	@source ../.venv/bin/activate && \
	for cfg in \
		$(SRC_DIR)/configs/sf_xl/intra_inter_cell10_cos02_sg16.yaml \
		$(SRC_DIR)/configs/sf_xl/intra_inter_cell10_cos02_sg32.yaml \
		$(SRC_DIR)/configs/sf_xl/intra_inter_cell10_cos02_sg64.yaml \
	; do \
		while [ $$(jobs -r | wc -l) -ge 4 ]; do sleep 10; done; \
		echo "--- Starting $$cfg ---" && python -m cli train $$cfg & \
	done; \
	wait && echo "All training runs complete"