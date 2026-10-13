<#
.SYNOPSIS
    PowerShell task runner for sim2real-robustness (Windows make alternative).
.EXAMPLE
    .\make.ps1 test
    .\make.ps1 robust-quick
    .\make.ps1 robust-all -RJOBS 4
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Target = "test",

    [string]$ENV = "HalfCheetah-v5",
    [int]$SEEDS = 5,
    [int]$STEPS = 1000000,
    [int]$JOBS = 1,

    [string]$RENV = "HalfCheetah-v5",
    [int]$RSEEDS = 5,
    [int]$RSTEPS = 1000000,
    [int]$RJOBS = 1,
    [string]$RDIR = "",
    [string]$RRES = "",
    [string]$MODEL = "runs/smoke/model.pt"
)

$ErrorActionPreference = "Stop"

# Locate Python in virtualenv or system
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = "python"
}

if (-not $RDIR) { $RDIR = "runs/robust_$RENV" }
if (-not $RRES) { $RRES = "results/robust_$RENV" }

switch ($Target.ToLower()) {
    "test" {
        & $py -m pytest -q
    }
    "lint" {
        & $py -m ruff check .
        & $py -m ruff format --check .
    }
    "fmt" {
        & $py -m ruff check --fix .
        & $py -m ruff format .
    }
    "smoke" {
        & $py -m robust_rl.train --env-id InvertedPendulum-v5 --total-timesteps 40960 --log-dir runs/smoke --eval-episodes 5
    }
    "train" {
        & $py -m robust_rl.train --env-id $ENV --total-timesteps $STEPS --seed 1 --log-dir "runs/${ENV}_seed1"
    }
    "ablate-quick" {
        & $py -m robust_rl.ablate --study tiny --env-id InvertedPendulum-v5 --seeds 3 --total-timesteps 40960 --eval-episodes 5 --out runs/quick
        & $py -m robust_rl.analysis --runs runs/quick --out results/quick
    }
    "robust-quick" {
        & $py -m robust_rl.robust_study --study tiny --env-id InvertedPendulum-v5 --seeds 3 --total-timesteps 61440 --out runs/robust_quick
        & $py -m robust_rl.robust_eval --runs runs/robust_quick --out results/robust_quick/eval.csv --episodes 3 --quick
        & $py -m robust_rl.robust_report --eval results/robust_quick/eval.csv --runs runs/robust_quick --out results/robust_quick
    }
    "robust-train" {
        & $py -m robust_rl.robust_study --study core --env-id $RENV --seeds $RSEEDS --total-timesteps $RSTEPS --jobs $RJOBS --out $RDIR
    }
    "robust-eval" {
        & $py -m robust_rl.robust_eval --runs $RDIR --out "$RRES/eval.csv" --episodes 10 --jobs $RJOBS
    }
    "robust-report" {
        & $py -m robust_rl.robust_report --eval "$RRES/eval.csv" --runs $RDIR --out $RRES
    }
    "robust-all" {
        & $py -m robust_rl.robust_study --study core --env-id $RENV --seeds $RSEEDS --total-timesteps $RSTEPS --jobs $RJOBS --out $RDIR
        & $py -m robust_rl.robust_eval --runs $RDIR --out "$RRES/eval.csv" --episodes 10 --jobs $RJOBS
        & $py -m robust_rl.robust_report --eval "$RRES/eval.csv" --runs $RDIR --out $RRES
    }
    "export" {
        & $py -m robust_rl.export --model $MODEL --out-dir export
    }
    "serve" {
        $env:PPO_MODEL_PATH = $MODEL
        & $py -m uvicorn --factory robust_rl.serve:create_app --host 0.0.0.0 --port 8000
    }
    "closed-loop" {
        & $py -m robust_rl.closed_loop --url http://localhost:8000 --env-id $RENV --axis gain --value 1.5 --episodes 3
    }
    default {
        Write-Error "Unknown target '$Target'. Available targets: test, lint, fmt, smoke, train, robust-quick, robust-train, robust-eval, robust-report, robust-all, export, serve, closed-loop"
    }
}
