openvalidators_dataset:
	# Downloads all the runs from the project and exports them to a csv file
	python3 data_collector.py \
				--download_all \
				--export_path='openvalidators_dataset.csv' \

run_id_dataset:
	# Downloads a specific run from the project and exports it to a csv file
	python3 data_collector.py \
				--export_path='$(RUN_ID)_openvalidators_dataset.csv' \
 				--wandb_run_id=$(RUN_ID) \

mining_dataset:
	# Downloads all the runs from the project with a mining dataset
	python3 data_collector.py  \
				--download_all \
				--export_path='openvalidators_dataset.csv' \
				--export_mining_dataset \

scored_mining_dataset:
	# Downloads all the runs from the project with a scored mining dataset
	python3 data_collector.py  \
			--download_all \
			--export_path='openvalidators_dataset.csv' \
			--export_mining_with_scoring_dataset \

openai_mining_dataset:
	python3 data_collector.py  \
		--download_all \
		--export_path='openvalidators_dataset.csv' \
		--export_openai_dataset \
