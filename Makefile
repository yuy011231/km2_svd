ifndef TARGET_FILES
	export TARGET_FILES:=km2_svd
endif
.PHONV: format
format:
	poetry run black ${TARGET_FILES}

.PHONV: lint
lint:
	poetry run ruff ${TARGET_FILES}

.PHONV: install
install:
	poetry install
