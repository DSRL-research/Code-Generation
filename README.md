# Code-on-demand sandbox

A caller sends a **chain**: a JSON program made of operators from a fixed list. The service fetches its sources
(links), applies the operators, and returns the result rows. The operators reshape rows (stream) or train, use and
compare models (ML). The caller never sends code: the expressions inside operators are single Python expressions,
checked against an allowed list before anything runs.

When a workflow step needs a capability that a microgrid lacks (a forecast, a model), an orchestrator can have a
chain generated for it on the spot, have it checked by the sandbox, and reuse it for the next microgrids. The tests
show this for UC1 and UC3 with 1, 2 and 3 microgrids.

**[DESIGN.md](DESIGN.md)** is the reference: the chain format, every operator, the expressions, the policy, the
ML operators (section 11), the use cases in the tests (section 16), and why each choice was made.

## Run

```bash
python3 -m venv venv
venv/bin/pip install -r requirements-dev.txt

# a data source: the data service of the Java runtime, on port 8000
python3 ../../edc-connectors/chain-runtime/data-service/service.py &

# the sandbox, on port 8095
venv/bin/uvicorn src.main:app --port 8095

venv/bin/pytest            # tests; they start their own source server
venv/bin/ruff check .      # lint
venv/bin/ruff format .     # format
```

Settings are `SANDBOX_<NAME>` variables, or a `.env` file (see `.env.example`). The policy check is switched off by
default; `SANDBOX_POLICY_ENABLED=true` switches it on, with the owner's rules from `policy.json`. Trained models go
to `SANDBOX_MODEL_DIR` (default `models/`).

## Use

```bash
# train a demand model for MG1, then forecast the next 4 hours with it
curl -X POST localhost:8095/chains/run -H 'content-type: application/json' -d '
{"source": "http://localhost:8000/microgrids/1/readings",
 "steps": [
   {"op": "reduce", "window": "1h", "aggregate": {"kw": "mean"}},
   {"op": "fit", "model": "mg1_demand", "target": "kw", "lags": [1, 2, 24], "holdout": 24}]}'

curl -X POST localhost:8095/chains/run -H 'content-type: application/json' -d '
{"source": "http://localhost:8000/microgrids/1/readings",
 "steps": [
   {"op": "reduce", "window": "1h", "aggregate": {"kw": "mean"}},
   {"op": "forecast", "model": "mg1_demand", "steps": 4, "as": "load"}]}'
```

Share a model with another sandbox. For a try on one machine, start a second sandbox with its own model store:

```bash
SANDBOX_MODEL_DIR=models_node2 venv/bin/uvicorn src.main:app --port 8096

# the second sandbox loads mg1_demand from the first one by its link, and uses it in the same request
curl -X POST localhost:8096/chains/run -H 'content-type: application/json' -d '
{"source": "http://localhost:8000/microgrids/2/readings",
 "steps": [
   {"op": "reduce", "window": "1h", "aggregate": {"kw": "mean"}},
   {"op": "load", "from": {"source": "http://localhost:8095/models/mg1_demand/export"}, "as": "mg1_demand"},
   {"op": "predict", "model": "mg1_demand", "as": "kw_pred"},
   {"op": "evaluate", "actual": "kw", "predicted": "kw_pred"}]}'
```

| Endpoint | Reply |
|---|---|
| `POST /chains/run` | 200: the rows |
| `POST /chains/check` | 200: the links the chain would fetch, and which are raw. Nothing is fetched |
| `GET /operators` | the operator catalogue: each operator, its family, what it does, the schema of its parameters |
| `GET /models`, `GET /models/{name}` | the cards of the trained models; `export` is the link to share a model by |
| `GET /models/{name}/export` | the model as one record of plain numbers: the source for `load` in another sandbox (linear models) |
| `GET /health` | 200 |

Errors: 400 `{"error": ...}` (the chain cannot be read or run), 403 `{"rejected": ...}` (the policy, when it is
switched on), 502 `{"error": ...}` (a source cannot be reached). Each error names the step, e.g.
`steps[1].with.steps[0] reduce`.

| Family | Operators |
|---|---|
| stream | `filter`, `map`, `reduce`, `join`, `union`, `top`, `lag`, `collect` |
| ml | `fit`, `train`, `prepare`, `finetune`, `load`, `predict`, `forecast`, `evaluate`, `embed`, `similarity` |

## Code

```
src/
  main.py, dependencies.py     the FastAPI app and its wiring
  config/                      settings; services.py puts the parts together (the policy check, or NoPolicy)
  routers/                     /chains/run, /chains/check, /operators, /models, /health; errors.py: errors -> replies
  services/                    ChainService (parse, check, fetch, run), ChainParser, builders/ (one per operator),
                               the catalogue, the reply
  models/                      the chain JSON (chain_json.py, steps/stream.py, steps/ml.py), Chain, Rows, Timestamp,
                               ModelRecord (a shared model)
  expressions/                 Expression: allowed lists, checker, rewriter, guards
  operations/stream/           filter, map, reduce/ (a package), join, union, top, lag, collect
  operations/ml/               fit, train, prepare, finetune, load, predict, forecast, evaluate, embed, similarity
  ml/                          features, algorithms, trained models, metrics, profiles, vectors, sharing
  store/                       the model store
  clients/                     SourceClient: GET a link, records -> rows
  policy/                      policy.json: source patterns, operator rules, PolicyChecker, NoPolicy
tests/
  test_*.py                    one file per area; test_uc1_workflow.py and test_uc3_adaptation.py for the use cases
  scenario/                    stand-ins for the orchestrator, its knowledge base and the LLM (not part of the sandbox)
  resources/                   real hourly readings of MG1, MG2, MG3, MG11, MG12; recipes; policies; small data sets
```

Each module is small and has one job. A request goes:
`routers/chains.py` → `ChainService.run` → `ChainParser.parse` (with the builders) → the policy check →
`Chain.execute` (→ `SourceClient.fetch` once per link, then each operation's `apply`) → `reply.to_reply`.

## Extend

**An operator:**

1. `models/steps/stream.py` or `ml.py`: its JSON model; add it to `StepJson` in `models/chain_json.py`.
2. `operations/stream/` or `operations/ml/`: a frozen dataclass that subclasses `Operation` and implements
   `apply(rows, ctx)`; add it to the family's list in that package's `__init__.py`.
3. `services/builders/stream.py` or `ml.py`: a builder method, and the pair in `builders()`.
4. `policy.json`: a rule for it (with the policy switched on, an operator without a rule is refused). For its own
   kind of rule (allowed algorithms, parameter ranges), write an `OperatorRule` in `policy/rules.py`.
5. `tests/`: its tests. `GET /operators` lists it from then on, so a code generator can use it.

Anything an operation needs from the run goes in `RunContext` (`operations/base.py`): the fetch-once cache and the
model store are there.

**An algorithm for `fit`:** add an entry to `ALGORITHMS` in `ml/algorithms.py`, with its parameters and ranges.
To make its models shareable, give it a `format` in `models/model_record.py`, and teach `ml/sharing.py` to write
and read it.

**A function or method in expressions:** add it to `FUNCTIONS` or `METHODS` in `expressions/allowed.py`, and to
the table in DESIGN.md section 6. Only add methods that are harmless on every value an expression can reach.

**A reduce criterion:** add a function to `operations/stream/reduce/criteria.py`.

**A capability in the use-case tests:** add the chain a generator should write to `tests/resources/recipes/`.
