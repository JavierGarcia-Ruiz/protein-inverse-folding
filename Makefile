.PHONY: install data train figures test notebooks clean

install:
	pip install -e .

data:
	python scripts/download_data.py

train:
	python scripts/train.py --config configs/default.yaml

figures:
	python scripts/make_figures.py

notebooks:
	python scripts/generate_notebooks.py

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

