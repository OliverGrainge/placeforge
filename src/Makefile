
SHELL := /bin/bash


train_datapipelines:
	@source ../.venv/bin/activate && \
	for pipeline in \
		sf_xl_cosplace \
		sf_xl_curavpr \
		sf_xl_eigenplaces \
		sf_xl_small_cosplace \
		sf_xl_small_curavpr \
		sf_xl_small_eigenplaces \
	; do \
		echo "--- $$pipeline ---" && python -m cli datapipeline $$pipeline; \
	done


val_datapipelines:
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


test_datapipelines:
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