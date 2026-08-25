.PHONY: install install-postgres install-demo test test-demo lint typecheck typecheck-demo run-scenarios run-demo grade-local clean

install:
	python -m pip install -e '.[dev,sqlite]'

install-postgres:
	python -m pip install -e '.[dev,postgres]'

install-demo:
	python -m pip install -e '.[dev,openai,sqlite,demo]'

test:
	pytest

test-demo:
	pytest demo/tests

lint:
	ruff check src tests demo

typecheck:
	mypy src

typecheck-demo:
	mypy src demo/streamlit_app.py

run-scenarios:
	python -m langgraph_agent_lab.cli run-scenarios --config configs/lab.yaml --output outputs/metrics.json

run-demo:
	python -m streamlit run demo/streamlit_app.py

grade-local:
	python -m langgraph_agent_lab.cli validate-metrics --metrics outputs/metrics.json

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov dist build *.egg-info outputs/*.json
