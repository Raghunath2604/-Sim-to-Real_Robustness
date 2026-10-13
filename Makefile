.PHONY: robust-quick robust-train robust-eval robust-report robust-all closed-loop install install-cpu test lint fmt smoke train ablate-quick ablate-core ablate-full analyze export serve docker-build docker-train clean

PY ?= python
ENV ?= HalfCheetah-v5
SEEDS ?= 5
STEPS ?= 1000000

install:
	$(PY) -m pip install -e ".[dev]"

# CPU-only PyTorch (much smaller download than the default CUDA build), then the project
install-cpu:
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	ruff check . && ruff format --check .

fmt:
	ruff check --fix . && ruff format .

# ~1 minute: proves the algorithm learns (InvertedPendulum reaches ~1000 return)
smoke:
	$(PY) -m robust_rl.train --env-id InvertedPendulum-v5 --total-timesteps 40960 --log-dir runs/smoke --eval-episodes 5

# one full single-seed run
train:
	$(PY) -m robust_rl.train --env-id $(ENV) --total-timesteps $(STEPS) --seed 1 --log-dir runs/$(ENV)_seed1

# fast end-to-end check of the whole ablation + analysis pipeline (a few minutes)
ablate-quick:
	$(PY) -m robust_rl.ablate --study tiny --env-id InvertedPendulum-v5 --seeds 3 --total-timesteps 40960 --eval-episodes 5 --out runs/quick
	$(PY) -m robust_rl.analysis --runs runs/quick --out results/quick

# the real study: 5 variants x SEEDS seeds on ENV (hours on a laptop CPU; use JOBS to parallelise)
JOBS ?= 1
ablate-core:
	$(PY) -m robust_rl.ablate --study core --env-id $(ENV) --seeds $(SEEDS) --total-timesteps $(STEPS) --jobs $(JOBS) --out runs/core_$(ENV)

ablate-full:
	$(PY) -m robust_rl.ablate --study full --env-id $(ENV) --seeds $(SEEDS) --total-timesteps $(STEPS) --jobs $(JOBS) --out runs/full_$(ENV)

analyze:
	$(PY) -m robust_rl.analysis --runs runs/core_$(ENV) --out results/core_$(ENV)

MODEL ?= runs/smoke/model.pt
export:
	$(PY) -m robust_rl.export --model $(MODEL) --out-dir export

serve:
	PPO_MODEL_PATH=$(MODEL) $(PY) -m uvicorn --factory robust_rl.serve:create_app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t sim2real .

docker-train:
	docker run --rm -v $(PWD)/runs:/app/runs sim2real python -m robust_rl.train --env-id InvertedPendulum-v5 --total-timesteps 40960 --log-dir runs/docker_smoke

clean:
	rm -rf runs results export .pytest_cache .ruff_cache

# ---------------- project 2: robustness study ----------------
RENV ?= HalfCheetah-v5
RSEEDS ?= 5
RSTEPS ?= 1000000
RJOBS ?= 1
RDIR ?= runs/robust_$(RENV)
RRES ?= results/robust_$(RENV)

# pipeline check on an easy task: baseline vs dr_0.5, 3 seeds, tiny grid (~10-15 minutes)
robust-quick:
	$(PY) -m robust_rl.robust_study --study tiny --env-id InvertedPendulum-v5 --seeds 3 --total-timesteps 61440 --out runs/robust_quick
	$(PY) -m robust_rl.robust_eval --runs runs/robust_quick --out results/robust_quick/eval.csv --episodes 3 --quick
	$(PY) -m robust_rl.robust_report --eval results/robust_quick/eval.csv --runs runs/robust_quick --out results/robust_quick

robust-train:
	$(PY) -m robust_rl.robust_study --study core --env-id $(RENV) --seeds $(RSEEDS) --total-timesteps $(RSTEPS) --jobs $(RJOBS) --out $(RDIR)

robust-eval:
	$(PY) -m robust_rl.robust_eval --runs $(RDIR) --out $(RRES)/eval.csv --episodes 10 --jobs $(RJOBS)

robust-report:
	$(PY) -m robust_rl.robust_report --eval $(RRES)/eval.csv --runs $(RDIR) --out $(RRES)

robust-all: robust-train robust-eval robust-report

# closed-loop test of a served policy (start `make serve MODEL=...` first)
closed-loop:
	$(PY) -m robust_rl.closed_loop --url http://localhost:8000 --env-id $(RENV) --axis gain --value 1.5 --episodes 3
