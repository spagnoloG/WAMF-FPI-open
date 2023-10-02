push-ml-node:
	@echo "Pushing code to ml-node"
	rsync -rav --progress \
		--exclude='.venv/' \
		--exclude='.git/' \
		--exclude='satellite_dataset/' \
		--exclude='__pycache__/' \
		--exclude='drone_dataset' \
		--exclude="code/cross-view-localization-model/**" \
		--exclude="code/__pycache__/**" \
		--exclude="code/wandb/**" \
		--include='code/**' \
		--include='*/' \
		--include='*.py' \
		--include='*.csv' \
		--include="*.txt" \
		--exclude='*' \
		. ml-node:/home/ml-node/Documents/uav-localization-experiments
