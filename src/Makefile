
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
		msls_cossim_0p00_train \
		msls_cossim_0p20_train \
		msls_cossim_0p40_train \
		sf_xl_small_cossim_0p00_train \
		sf_xl_small_cossim_0p20_train \
		sf_xl_small_cossim_0p40_train \
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

