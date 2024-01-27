install:
	pip install --upgrade pip && pip install -r requirements/requirements.txt && pip install -r requirements/test_requirements.txt

train_pipeline:
	python bikeshare_model/train_pipeline.py

format:
	black . *.py

lint:
	pylint --disable=R,C *.py

mypy:
	mypy --implicit-optional bikeshare_model/ tests/

test:
	python -m pytest tests/test_*.py

all: install format lint mypy train_pipeline test
