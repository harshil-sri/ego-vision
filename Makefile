.PHONY: test smoke app equivalence check clean

test:
	python3 -m pytest tests/ -q --ignore=tests/equivalence

equivalence:
	python3 -m pytest tests/equivalence -q

smoke:
	python3 -m pytest tests/test_app_smoke.py -q

app:
	python3 -m scripts.run_app --mode keyboard

app-synthetic:
	python3 -m scripts.run_app --mode keyboard --synthetic

check:
	bash scripts/check_all.sh

clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null; true
