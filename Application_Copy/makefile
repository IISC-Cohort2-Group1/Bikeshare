#makefile

install:
		pip install --upgrade pip &&\
				pip install -r /home/ec2-user/environment/Bikeshare/Application_Copy/requirements/requirements.txt

format:
		black *.py

lint:
		pylint --disable=R,C /home/ec2-user/environment/Bikeshare/Application_Copy/bikeshare_model/train_pipeline.py

test:
		python -m pytest tests/test_*.py

all : install lint test format