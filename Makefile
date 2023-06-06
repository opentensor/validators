project_dataset:
	# Downloads all the runs from the project and exports them to a csv file
	python3 scripts/download_wandb_run_logs.py --download_all --export_path='project_dataset.csv'

run_id_dataset:
	# Downloads a specific run from the project and exports it to a csv file
	python3 scripts/download_wandb_run_logs.py  --export_path='openvalidators_dataset.csv' --wandb_run_id=$(RUN_ID)

mining_dataset:
	# Downloads all the runs from the project with a mining dataset
	python3 scripts/download_wandb_run_logs.py  --download_all --export_path='openvalidators_dataset.csv' --export_mining_dataset