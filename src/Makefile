
SHELL := /bin/bash


datapipelines_sf_xl:
	@source ../.venv/bin/activate && \
	echo "--- sf_xl_cosplace ---" && python -m cli datapipeline sf_xl_cosplace && \
	echo "--- sf_xl_eigenplaces ---" && python -m cli datapipeline sf_xl_eigenplaces && \
	echo "--- sf_xl_curavpr ---" && python -m cli datapipeline sf_xl_curavpr


datapipelines_sf_xl_small:
	@source ../.venv/bin/activate && \
	echo "--- sf_xl_small_cosplace ---" && python -m cli datapipeline sf_xl_small_cosplace && \
	echo "--- sf_xl_small_eigenplaces ---" && python -m cli datapipeline sf_xl_small_eigenplaces && \
	echo "--- sf_xl_small_curavpr ---" && python -m cli datapipeline sf_xl_small_curavpr




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


datapipelines_train: train_datapipelines_sf_xl train_datapipelines_sf_xl_small



analyse_sf_xl:
	@source ../.venv/bin/activate && \
	echo "--- sf_xl_cosplace ---" && python -m cli analyse sf_xl_cosplace && \
	echo "--- sf_xl_eigenplaces ---" && python -m cli analyse sf_xl_eigenplaces && \
	echo "--- sf_xl_curavpr ---" && python -m cli analyse sf_xl_curavpr


analyse_sf_xl_small:
	@source ../.venv/bin/activate && \
	echo "--- sf_xl_small_cosplace ---" && python -m cli analyse sf_xl_small_cosplace && \
	echo "--- sf_xl_small_eigenplaces ---" && python -m cli analyse sf_xl_small_eigenplaces && \
	echo "--- sf_xl_small_curavpr ---" && python -m cli analyse sf_xl_small_curavpr


analyse: analyse_sf_xl analyse_sf_xl_small


