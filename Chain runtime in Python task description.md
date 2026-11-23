# Chain runtime in Python: task description

Nov 20, 2026 · @vof

## Goal and first scope

Rewrite the chain runtime (Java 21, Spring Boot, about 1,000 lines) in Python. It keeps the same chain JSON, replies and policy, and then gains ML operators. Python because the ML libraries are native there, and the data service, `local_agg.py` and `plan-orch` are already Python.

| Version | What it does | Done when |
| --- | --- | --- |
| v1: parity | Same chain JSON with URL sources, the 7 stream operators, the expression language, the raw-readings policy, `POST /chains/run` and `GET /chains/{name}`, the same 200 / 400 / 403 replies | Every Java test case and every documented example returns the same rows from both runtimes |
| v2: first ML operators | `featurize`, `fit`, `predict`; a local model store; policy rules for the three | A model trained on MG\_1 forecasts the next hour, and its error is computed with stream operators |
| Later | `evaluate`, `embed`, model transfer by URL, `train_round` and `merge`, calls to the lab ML API as sources, registration of results as data assets | Separate tasks |

Out of scope for v1 and v2: the EDC connector, the catalog and asset registration, the LLM, authentication, and streaming. Sources are fetched whole, as now.

The Java runtime stays the reference until v1 passes; the parity tests run both.

## How the current runtime works

The runtime is a small interpreter. It parses the chain JSON into operator objects and checks them against `policy.json` without running anything. Then it fetches each source link once and applies the steps to a list of rows.

1. `POST /chains/run` receives a chain; `GET /chains/{name}` runs `resources/chains/{name}.json`.
2. `ChainParser` builds a `Chain`: source URI, `raw` flag, steps, fetch function. `"MG_n"` becomes `<data-service.url>/microgrids/n/readings?start=&end=`; a link is used as it is. An unknown `op` is rejected with 403.
3. `PolicyChecker` walks the steps without fetching. While the rows are raw (from `"raw": true` until the first `reduce`), each operator must follow its `onRaw` rule. Chains under `with` are checked too and must not output raw rows. A chain whose output is still raw is rejected.
4. `Chain.execute` fetches the source (once per link per run) and applies the steps in order. `join` and `union` run their `with` chain when they are reached.
5. Reply: 200 with the rows, 400 `{"error": ...}`, or 403 `{"rejected": "<op>: <reason>"}`.

**Rows.** A row is field → value. From a source record, `timestamp` or `date` is kept as text and parsed into `ts`, in minutes since 2026-06-01 00:00 with any offset ignored. Integers and decimals are kept; text, booleans, lists and objects are dropped. `day`, `weekday`, `hour` and `slot` are derived from `ts` when read.

**Expressions** (in `where` and `set`): numbers, unquoted times, field names, `+ - * / %`, comparisons, `&& || !`, `c ? a : b`. There are no functions and no strings. Each expression is parsed once and evaluated per row; its field names feed the policy check.

| Operator | JSON | Behaviour as implemented |
| --- | --- | --- |
| `filter` | `where` | Keeps the rows where the expression is true |
| `map` | `set` | Sets fields in order; a later expression reads earlier ones. A comparison stores true/false, and later reads of that field fail |
| `reduce` | `window` or `by`; `criteria`; `field` | `window`: tumbling windows aligned to midnight, key `ts` = window start; a window is kept only if it holds (window / smallest gap) readings. `by`: groups on whole-number keys. Output: the keys plus the one field, sorted by key |
| `join` | `on`, `with` | Inner join; the right side must have one row per key; the left value wins on a name clash; keys compared as whole numbers |
| `union` | `with` | Appends the right rows; duplicates kept |
| `top` | `n`, `by` | The n rows with the highest value; ties keep input order |
| `collect` | `fields`, `decimals` (default 2) | Writes `ts` as a time, text as it came, whole numbers as integers, others rounded half up |

**Policy** (`policy.json`): time fields `ts`, `day`, `hour`, `slot`. On raw rows, `filter` may use time fields only. `reduce` needs a window of at least 15 minutes with mean, sum, count or max. `map`, `join`, `union`, `top` and `collect` are not allowed on raw rows.

**Size:** 1,012 lines of main code in 20 files; 476 lines of tests (`ChainServiceTest`, `ExprTest`, `ChainControllerTest`); Spring Boot 3.3.5.

### Found in the code, to settle before porting

- **The chain decides what is raw.** Leave out `"raw": true` and MG\_n readings come back unchanged: test `rawIsFalseWhenLeftOut` returns 288 raw readings. The owner should decide this, not the caller.
- **max on raw.** `policy.json` allows it, but the README, the agent prompt and test `rejectsMaxOnReadings` forbid it. Reading the code, that test fails with the current `policy.json`.
- **weekday.** The agent prompt allows a filter on `weekday` over raw rows. `policy.json` leaves `weekday` out of the time fields, so such a filter is rejected.
- **Duplicate join keys** end in HTTP 500 with no message, not 400.
- **Any link is fetched.** A chain can make the runtime call any address it can reach.
- **Naming.** `ORCHESTRATOR.md` calls the runtime "the Orchestrator", which clashes with the Federated Orchestrator of the paper.

## Input: the chain JSON

The chain JSON does not change: v1 accepts every chain the Java runtime accepts. The source is a URL the runtime calls with GET; `MG_n` stays as a short form for the data service's readings.

| Key | Meaning |
| --- | --- |
| `source` | A link, `"http://localhost:8000/microgrids/17/demand_forecast"`, or the short form `"MG_1"` ... `"MG_20"` |
| `start`, `end` | Optional, short form only: first and last day, `"2026-07-24"`. For a link, put them in the link |
| `raw` | Optional, default false: the rows are raw readings, so the policy applies until the first `reduce` |
| `steps` | The operators, applied in order; `join` and `union` hold another chain of the same form under `with` |

A link source, as `ORCHESTRATOR.md` runs it:

```json
{"source": "http://localhost:8000/microgrids/17/demand_forecast",
 "steps": [{"op": "collect", "fields": ["ts", "load"], "decimals": 1}]}
```

Raw readings with a second source under `with`:

```json
{"source": "MG_1", "raw": true, "steps": [
  {"op": "filter", "where": "day == 59"},
  {"op": "reduce", "window": "15m", "criteria": "mean", "field": "kw"},
  {"op": "map", "set": {"forecast_kw": "kw"}},
  {"op": "join", "on": "slot", "with": {"source": "MG_1", "raw": true, "steps": [
    {"op": "filter", "where": "day < 59"},
    {"op": "reduce", "window": "15m", "criteria": "mean", "field": "kw"},
    {"op": "reduce", "by": "slot", "criteria": "mean", "field": "kw"},
    {"op": "map", "set": {"baseline_kw": "kw"}}]}},
  {"op": "map", "set": {"flex_kw": "forecast_kw - baseline_kw"}},
  {"op": "collect", "fields": ["slot", "forecast_kw", "baseline_kw", "flex_kw"], "decimals": 1}]}
```

**What a link must reply:** a JSON list of records, or an object with the list under `readings`. Per record, `timestamp` or `date` becomes `ts`, numbers are kept, everything else is dropped. A 4xx reply fails the chain with 400 and the service's `error` text.

### Small additions (v1, backward compatible)

- **The owner marks raw sources.** `policy.json` gets a `sources` list, for example `[{"match": "^http://localhost:8000/microgrids/\\d+/readings", "raw": true}]`. A chain may set `raw` to true itself, but it cannot turn it off for a matching link. This closes the bypass found in the code.
- **The owner limits links.** The settings get a list of allowed hosts, a timeout per fetch and a maximum number of records per source.

### Later, not v1: sources that are calls

The lab ML API is called with POST and can answer through a job (`POST /api/v1/ops/{op}?mode=async`). A source object would cover that without changing the string form:

```json
{"source": {"url": "http://192.168.201.114:8093/api/v1/ops/predict", "method": "POST",
            "body": {"task": "demand", "horizon": 16}, "records": "result"}}
```

`records` names where the list sits in the reply. The body fields above are placeholders until the lab API's reply shape is probed.

## Operators

The 7 stream operators move over unchanged: same names, parameters and behaviour as the table in the previous section. Three ML operators are added in v2, in the same JSON style, and none of them may run on raw rows.

### Porting the stream operators: where Python differs from Java

The same code written naively in Python gives different numbers in a few places. Each one needs a test.

| Java behaviour | Naive Python | Do this |
| --- | --- | --- |
| `%` takes the sign of the left side: -7 % 3 = -1 | `-7 % 3` = 2 | `math.fmod` |
| x / 0 gives Infinity or NaN | `ZeroDivisionError` | return `inf`, `-inf` or `nan` |
| `Math.round` rounds half up (2.25 → 2.3 with 1 decimal) | `round()` rounds half to even | `math.floor(v * 10**d + 0.5) / 10**d` |
| JSON `true` is not a number | `bool` is a subclass of `int` | drop booleans explicitly when reading records |
| Groups come out sorted by key (`TreeMap`) | `dict` keeps insertion order | sort groups by key |
| Keys are cast to whole numbers (`(long)`) |  | `int()` truncates the same way |
| Three time formats, offset ignored |  | `datetime.fromisoformat`, then drop `tzinfo` |
| Expressions are parsed by a small grammar | `eval()` would run any code | port the recursive-descent parser; never `eval` |

Rows stay a list of dicts in v1, not a pandas DataFrame. That keeps the behaviour of the Java rows and makes replies easy to diff. numpy is used only inside the ML operators.

### ML operators (v2)

| Operator | JSON | What it does | Output |
| --- | --- | --- | --- |
| `featurize` | `target`, `lags`, `calendar`, `future`; or `model` to copy a stored model's spec | For each row with `lags` earlier rows at the same step: adds `lag_1` ... `lag_n` (the target's value 1 ... n steps earlier) and the calendar fields (`hour`, `weekday`, ...). With `future: true`, adds one row for the step after the last one, without the target | rows |
| `fit` | `model` (a name), `algo` (`linear`, `gbr`), `params`, `holdout` (rows) | Trains on the rows' features and target, scores the last `holdout` rows (mean absolute error), saves the model and its card in the model store | the chain's exit: one row `{model, algo, target, features, rows, holdout_rows, mae}` |
| `predict` | `model`, `as` | Adds the field named in `as`: the stored model's prediction for each row | rows |

v2 forecasts one step ahead; longer horizons come later. Every row keeps its `ts`, so time fields, filters and joins work after the ML operators as before.

Train on days 0-52 of MG\_1, hourly (1,272 hours, 1,248 rows after 24 lags, the last 168 held out):

```json
{"source": "MG_1", "raw": true, "end": "2026-07-23", "steps": [
  {"op": "reduce", "window": "1h", "criteria": "mean", "field": "kw"},
  {"op": "featurize", "target": "kw", "lags": 24, "calendar": ["hour", "weekday"]},
  {"op": "fit", "model": "mg1_load_1h", "algo": "gbr",
   "params": {"max_depth": 3, "n_estimators": 200}, "holdout": 168}]}
```

Score it on days 53-59; the error is computed with stream operators only:

```json
{"source": "MG_1", "raw": true, "start": "2026-07-23", "steps": [
  {"op": "reduce", "window": "1h", "criteria": "mean", "field": "kw"},
  {"op": "featurize", "model": "mg1_load_1h"},
  {"op": "predict", "model": "mg1_load_1h", "as": "kw_pred"},
  {"op": "filter", "where": "day >= 53"},
  {"op": "map", "set": {"err": "kw - kw_pred", "abs_err": "err > 0 ? err : -err", "all": "1"}},
  {"op": "reduce", "by": "all", "criteria": "mean", "field": "abs_err"},
  {"op": "collect", "fields": ["abs_err"], "decimals": 1}]}
```

Forecast the first hour after the data (2026-07-31 00:00):

```json
{"source": "MG_1", "raw": true, "start": "2026-07-29", "steps": [
  {"op": "reduce", "window": "1h", "criteria": "mean", "field": "kw"},
  {"op": "featurize", "model": "mg1_load_1h", "future": true},
  {"op": "predict", "model": "mg1_load_1h", "as": "kw_pred"},
  {"op": "filter", "where": "ts >= 2026-07-31"},
  {"op": "collect", "fields": ["ts", "kw_pred"], "decimals": 1}]}
```

**Model store.** `models/<name>.joblib` plus `models/<name>.json`, the card: algo, params, target, features in order, lags, calendar, step in minutes, source and date range, rows, holdout MAE, time of training. `GET /models` and `GET /models/{name}` return cards, never the file. joblib files are local only: loading one from another participant can run arbitrary code. Model transfer needs a safe format (skops, ONNX), which comes later.

## Design

One Python package mirrors the Java classes one to one, so each module can be checked against the file it replaces.

&#91;embedded content: Python chain runtime · request flow and modules\]

The policy check reads only the parsed chain and `policy.json`; sources are fetched only after it passes. The model store (dashed) arrives with v2.

| Python module | Replaces | Holds |
| --- | --- | --- |
| `chainrt/app.py` | `controller/ChainController.java` | Endpoints; maps errors to 400 / 403 |
| `chainrt/service.py` | `service/ChainService.java` | Parse, check, run; the per-run fetch cache |
| `chainrt/parser.py` | `service/ChainParser.java` | JSON → `Chain` dataclass; key and type checks per operator |
| `chainrt/sources.py` | `client/SourceClient.java` | GET a link; records → rows; host, timeout and size limits |
| `chainrt/policy.py` | `policy/PolicyChecker.java` | Static checks from `policy.json`, ML rules included |
| `chainrt/rows.py`, `chainrt/timefmt.py` | `model/Row.java`, `model/Time.java` | Rows as dicts; derived time fields; `ts` ↔ text |
| `chainrt/expr.py` | `expression/Expr.java` | The expression parser and evaluator |
| `chainrt/ops/stream.py` | `operation/*.java` | The 7 stream operators |
| `chainrt/ops/ml.py`, `chainrt/models.py` | new (v2) | `featurize`, `fit`, `predict`; the model store |
| `tests/` | `src/test/` | pytest ports and the parity harness |

Every operator keeps the Java contract: it takes rows and returns new rows, never changing its input. `join` and `union` also hold the chain under `with`.

```python
class Operation(Protocol):
    name: str                                     # as in the JSON and in policy.json
    def apply(self, rows: list[dict]) -> list[dict]: ...
```

Dependencies: FastAPI and uvicorn, pytest; scikit-learn and joblib from v2. Settings come from one file, like `application.properties` now: the data service URL, allowed hosts, fetch timeout, record limit and the model folder.

## Validation and policy checks

Nothing is fetched until the chain has been parsed and passed the policy, as now; the Python runtime keeps that order and the same status codes.

| Stage | Checks | Reply on failure |
| --- | --- | --- |
| Parse | JSON shape; every operator's required keys and their types; expressions parse; `fit` only as the last step | 400 `{"error": ...}`; an unknown `op` stays 403, as in Java |
| Policy (static, no I/O) | `onRaw` rules until the first `reduce`; chains under `with` checked and not raw at output; output not raw; raw marking from the owner's `sources` rules; ML operator rules below | 403 `{"rejected": "<op>: <reason>"}` |
| Fetch | Allowed host, timeout, maximum records, reply shape | 400 with the link and the reason |
| Run | Unknown field, text where a number is needed, duplicate join key, `filter` that does not return true/false, unknown model, too few rows for `fit` | 400; never 500 |

Error texts may be worded differently from Java; the status code and the `<op>: <reason>` form must match.

### Policy rules for the ML operators (v2)

The same `policy.json`, with three new entries. `onRaw: false` for all three means a chain must `reduce` first, so no model ever sees raw readings.

```json
"featurize": {"onRaw": false, "maxLags": 168},
"fit":       {"onRaw": false, "minRows": 168,
              "algos": {"linear": {},
                        "gbr": {"max_depth": [1, 6], "n_estimators": [10, 500]}}},
"predict":   {"onRaw": false}
```

`algos` and the parameter ranges are checked statically, before anything is fetched. `minRows` can only be checked at run time, just before training; it still replies 403. These limits are placeholders.

This is the computational policy on ML operators that the professor asked for: the owner says which algorithms may run and with which settings, as `reduce` already says which windows and criteria are allowed.

## Work plan

Six milestones, each closed by tests. The first four only reproduce Java; ML starts at milestone 6, on a runtime already proven identical.

1. **Expressions and time.** Port `Expr` and `Time`.
   - Done when the `ExprTest` cases pass in pytest, plus one test per row of the porting table (`%`, division by zero, rounding, booleans, offsets).
2. **Sources and rows.** GET a link; expand `MG_n` with `start` and `end`; turn records into rows; derive the time fields; fetch each link once per run.
   - Tests start `data-service/service.py` on a free port, as the Java tests do (it prints `listening on ...`).
3. **Stream operators, policy and HTTP.** The 7 operators, the static policy check, `POST /chains/run`, `GET /chains/{name}` and the three bundled chains.
   - Done when every case of `ChainServiceTest` and `ChainControllerTest` passes in pytest with the same rows and status codes.
4. **Parity harness.** A script sends the same chains to both runtimes and diffs the replies: `resources/chains`, every example in `ORCHESTRATOR.md`, and every chain `plan-orch` builds. Then point `plan-orch/settings.json` (`orchestrator`) at the Python runtime and rerun `baseline` and `demand_avg`.
   - Done when every diff is empty and the new runs match `plan-orch/runs/`.
5. **The fixes.** Owner-side raw sources, allowed hosts with timeout and record limit, 400 on duplicate join keys, and the team's decision on `max` and `weekday` on raw rows.
   - Each fix with a test that fails on the Java runtime and passes on the Python one. From here the Java runtime is retired.
6. **ML operators.** `featurize`, `fit` (`linear`, `gbr`), `predict`, the model store and the three policy rules.
   - Done when the three chains of the Operators section run: training returns one card, scoring returns one error row, forecasting returns one row for 2026-07-31 00:00.
   - Also: `fit` on raw rows → 403; an algorithm or parameter outside `policy.json` → 403; an unknown model → 400.

Later, each as its own task: `evaluate` and `embed`; source objects for POST calls and async jobs (lab ML API); model transfer in a safe format; `train_round` and `merge`; registering outputs and pipelines as data assets in the connector.

## Open questions

- [ ] **Who marks raw sources?** The chain does now (`"raw": true`). Proposed: the owner, through `sources` rules in `policy.json`.
- [ ] **max on raw rows?** `policy.json` allows it; the README, the agent prompt and `rejectsMaxOnReadings` forbid it.
- [ ] **weekday on raw rows?** The agent prompt allows filters on it; `policy.json` does not list it as a time field.
- [ ] **HTTP framework.** FastAPI (proposed: request handling and error replies in a few lines) or the standard library, as the data service uses.
- [ ] **ML library.** scikit-learn for v2 (proposed). Fine-tuning with frozen layers (UC3) needs PyTorch later.
- [ ] **Model names.** Does `fit` overwrite a model of the same name, or keep versions (`mg1_load_1h@2`)?
- [ ] **Safe model format** for transfer between participants: skops or ONNX.
- [ ] **Lab ML API.** Reply shape and async jobs of `POST /api/v1/ops/{op}` must be probed before source objects are designed.
- [ ] **Name.** Call it the chain runtime everywhere, and rename `ORCHESTRATOR.md`, to keep "Orchestrator" for the Federated Orchestrator.
