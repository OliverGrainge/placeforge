
SHELL := /bin/bash


datapipelines_train:
	@source ../.venv/bin/activate && \
	echo "--- sf_xl_cosplace ---" && python -m cli datapipeline sf_xl_cosplace && \
	echo "--- sf_xl_eigenplaces ---" && python -m cli datapipeline sf_xl_eigenplaces && \
	echo "--- sf_xl_curavpr_con ---" && python -m cli datapipeline sf_xl_curavpr_con && \
	echo "--- sf_xl_curavpr_cls ---" && python -m cli datapipeline sf_xl_curavpr_con




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




