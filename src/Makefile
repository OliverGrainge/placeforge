
SHELL := /bin/bash

sf_xl_datapipeline:
	source ../.venv/bin/activate && \
	echo "--- sf_xl ---" && \
	python -m cli datapipeline sf_xl && \
	echo "--- sf_xl_cosplace ---" && \
	python -m cli datapipeline sf_xl_cosplace && \
	echo "--- sf_xl_eigenplaces ---" && \
	python -m cli datapipeline sf_xl_eigenplaces && \
	echo "--- sf_xl_intra ---" && \
	python -m cli datapipeline sf_xl_intra && \
	echo "--- sf_xl_inter ---" && \
	python -m cli datapipeline sf_xl_inter && \
	echo "--- sf_xl_intra_inter ---" && \
	python -m cli datapipeline sf_xl_intra_inter && \
	echo "--- sf_xl ---" && \
	python -m cli analyse sf_xl && \
	echo "--- sf_xl_cosplace ---" && \
	python -m cli analyse sf_xl_cosplace --image-embedding-name sf_xl --place-embedding-name sf_xl_cosplace && \
	echo "--- sf_xl_eigenplaces ---" && \
	python -m cli analyse sf_xl_eigenplaces --image-embedding-name sf_xl --place-embedding-name sf_xl_eigenplaces && \
	echo "--- sf_xl_intra ---" && \
	python -m cli analyse sf_xl_intra --image-embedding-name sf_xl && \
	echo "--- sf_xl_inter ---" && \
	python -m cli analyse sf_xl_inter --image-embedding-name sf_xl --place-embedding-name sf_xl_inter && \
	echo "--- sf_xl_intra_inter ---" && \
	python -m cli analyse sf_xl_intra_inter --image-embedding-name sf_xl --place-embedding-name sf_xl_intra_inter

sf_xl_small_datapipeline:
	source ../.venv/bin/activate && \
	echo "--- sf_xl_small ---" && \
	python -m cli datapipeline sf_xl_small && \
	echo "--- sf_xl_small_cosplace ---" && \
	python -m cli datapipeline sf_xl_small_cosplace && \
	echo "--- sf_xl_small_eigenplaces ---" && \
	python -m cli datapipeline sf_xl_small_eigenplaces && \
	echo "--- sf_xl_small_intra ---" && \
	python -m cli datapipeline sf_xl_small_intra && \
	echo "--- sf_xl_small_inter ---" && \
	python -m cli datapipeline sf_xl_small_inter && \
	echo "--- sf_xl_small_intra_inter ---" && \
	python -m cli datapipeline sf_xl_small_intra_inter && \
	echo "--- sf_xl_small ---" && \
	python -m cli analyse sf_xl_small && \
	echo "--- sf_xl_small_cosplace ---" && \
	python -m cli analyse sf_xl_small_cosplace --image-embedding-name sf_xl_small --place-embedding-name sf_xl_small_cosplace && \
	echo "--- sf_xl_small_eigenplaces ---" && \
	python -m cli analyse sf_xl_small_eigenplaces --image-embedding-name sf_xl_small --place-embedding-name sf_xl_small_eigenplaces && \
	echo "--- sf_xl_small_intra ---" && \
	python -m cli analyse sf_xl_small_intra --image-embedding-name sf_xl_small && \
	echo "--- sf_xl_small_inter ---" && \
	python -m cli analyse sf_xl_small_inter --image-embedding-name sf_xl_small --place-embedding-name sf_xl_small_inter && \
	echo "--- sf_xl_small_intra_inter ---" && \
	python -m cli analyse sf_xl_small_intra_inter --image-embedding-name sf_xl_small --place-embedding-name sf_xl_small_intra_inter



SRC_DIR := /home/oliver/github/placeforge/src

sf_xl_train:
	@tmux new-session -d -s train_eigenplaces -c $(SRC_DIR) 'source ../.venv/bin/activate && python -m cli train $(SRC_DIR)/configs/sf_xl/eigenplaces.yaml' && \
	tmux new-session -d -s train_cosplace -c $(SRC_DIR) 'source ../.venv/bin/activate && python -m cli train $(SRC_DIR)/configs/sf_xl/cosplace.yaml' && \
	tmux new-session -d -s train_intra -c $(SRC_DIR) 'source ../.venv/bin/activate && python -m cli train $(SRC_DIR)/configs/sf_xl/intra.yaml' && \
	tmux new-session -d -s train_intra_inter -c $(SRC_DIR) 'source ../.venv/bin/activate && python -m cli train $(SRC_DIR)/configs/sf_xl/intra_inter.yaml'
	@echo "Launched 5 sessions. Attach with: tmux attach -t train_cosplace"