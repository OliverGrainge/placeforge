
SHELL := /bin/bash


datapipelines_train:
	@source ../.venv/bin/activate && \
	for pipeline in \
		gsvcities_train \
		msls_gsvcities_train \
		msls_train \
		sf_xl_gsvcities_train \
		sf_xl_small_gsvcities_train \
		sf_xl_small_train \
		sf_xl_train \
	; do \
		echo "--- $$pipeline ---" && python -m cli datapipeline $$pipeline; \
	done


datapipelines_train_ablation:
	@source ../.venv/bin/activate && \
	for pipeline in \
		sf_xl_abl_raw_gps \
		sf_xl_abl_sg_only \
		sf_xl_abl_filter_only \
		sf_xl_abl_tau02 \
		sf_xl_abl_tau04 \
		sf_xl_abl_sg512 \
		sf_xl_abl_sg2048 \
	; do \
		echo "--- $$pipeline ---" && python -m cli datapipeline $$pipeline; \
	done




datapipelines_val:
	@source ../.venv/bin/activate && \
	for pipeline in \
		msls_val \
		pitts30k_val \
		sf_xl_small_val \
		sf_xl_val \
		svox_val \
	; do \
		echo "--- $$pipeline ---" && python -m cli datapipeline $$pipeline; \
	done


datapipelines_test:
	@source ../.venv/bin/activate && \
	for pipeline in \
		nordland_test \
		pitts30k_test \
		sf_xl_small_test \
		sf_xl_test \
		svox_test \
		tokyo247_test \
	; do \
		echo "--- $$pipeline ---" && python -m cli datapipeline $$pipeline; \
	done

