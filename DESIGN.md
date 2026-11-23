# Code-on-demand sandbox: design

v1 (stream operators) and v2 (ML operators, use-case tests) · 2026-11-20 · how to run it: README.md

## 1. Purpose and scope

A standalone HTTP service. A caller sends a **chain**: a JSON program made of operators from a fixed list. The
service checks the chain against the data owner's **policy**, fetches the chain's **sources** (links), applies the
operators, and returns the result rows.

The caller never sends code. The expressions inside operators are single Python expressions, checked against an
allowed list before they run (section 6). This is the sandbox: only the listed operators, functions and methods
can run, and the policy is checked before anything is fetched. Later, the connector will run this service as a
process, with its own process limits.

**v1 contains:**

- the chain format, with parse and validation;
- sources: links that the service calls with GET;
- 7 stream operators: `filter`, `map`, `reduce`, `join`, `union`, `top`, `collect`;
- the expressions;
- the policy: allowed links, raw links, and the rules for each operator on raw rows;
- `POST /chains/check`, `POST /chains/run` and `GET /health`, all synchronous.

**v2 adds** (sections 11 and 16):

- 7 ML operators: `fit`, `finetune`, `predict`, `forecast`, `evaluate`, `embed`, `similarity`;
- a model store, with the models as assets: `GET /models`, `GET /models/{name}`;
- the operator catalogue that a code generator works from: `GET /operators`;
- the policy check switched off by default (`SANDBOX_POLICY_ENABLED=true` switches it back on; the code stays);
- tests of UC1 and UC3 with 1, 2 and 3 microgrids, with a stand-in for the orchestrator, its knowledge base and
  the LLM.

**v2.1 adds** (section 11, "Sharing models"):

- the `load` operator, and `GET /models/{name}/export`: models move between sandboxes by link, as plain numbers.

**Not yet:** the EDC connector, authentication, async jobs, streaming, and sharing `gbr` models.

**No compatibility target.** This is a new project. The Java runtimes (`edc-connectors/chain-runtime/` and the
connector filter) are only a reference for what the operators do. Compatibility with `plan-orch` is not a goal.
Section 12 lists each place where this design differs from Java, and why.

## 2. Request flow

```
POST /chains/run  {chain}
  1. parse    JSON -> Chain; every expression is parsed and checked      fails -> 400
  2. check    Chain + policy.json; reads nothing over the network       fails -> 403
  3. fetch    each link once (GET); records -> rows                     fails -> 400 or 502
  4. execute  the steps in order; join and union run their "with" chain fails -> 400
  5. reply    200 [rows]
```

Nothing is fetched until steps 1 and 2 pass. `POST /chains/check` does steps 1 and 2 only.

## 3. Chain format

```json
{"source": "http://localhost:8000/microgrids/2/battery_flex",
 "steps": [
   {"op": "filter", "where": "ts.between('2020-02-25T13:00:00', '2020-02-25T14:00:00')"},
   {"op": "reduce", "aggregate": {"charge*": "sum", "discharge*": "sum"}},
   {"op": "collect", "decimals": 2}]}
```

| Key | Meaning |
|---|---|
| `source` | A link. The service calls it with GET. Dates and other parameters are part of the link. |
| `steps` | The operators, applied in order. Optional; the default is `[]`. |

- A chain under `with` (in `join` and `union`) has the same form, with its own `source`.
- An unknown key is an error (400). A typo such as `"feilds"` is reported, not ignored.
- The chain has no `raw` key. The owner's policy says which links give raw rows (section 8).
- The format is defined by pydantic models, one per operator. Later, the same models can publish a JSON schema for
  the LLM prompt.
- Later, not v1: `source` can also be an object, for calls with POST (the lab ML API), or for a time that is not
  ISO text, e.g. `{"url": "...", "method": "POST", "body": {...}, "time": {"field": "ts", "format": "%d/%m/%Y %H:%M"}}`.

## 4. Sources, rows and time

### What a link must reply

JSON, in one of three forms:

- a list of records;
- an object with the list under `readings`;
- an object with exactly one field that holds a list.

| Problem with a source | Reply |
|---|---|
| The link replies 4xx | 400, with the source's `error` text |
| The reply is not JSON, or holds no list of records | 400 |
| More records or bytes than the limits allow (section 9) | 400: make the link narrower |
| The link cannot be reached, replies 5xx, or times out | 502 |

Redirects are not followed, because a redirect could leave the allowed links. Each link is fetched once per run,
also when several chains under `with` use it.

### Records become rows

- A row is a Python `dict`. **Every field of the record is kept as it came**: numbers, text, `true`/`false`,
  `null`, lists and objects. Java kept only the numbers.
- The time of a record: the first of `ts`, `timestamp`, `time` and `date` that holds ISO text is read into `ts`.
  The list of names is a setting.
  - If the text was in `ts`, the time value replaces it.
  - If it was in `timestamp`, `time` or `date`, that text field stays in the row as it came.
- A record without such a field has no `ts`. A `reduce` with a `window` cannot be used on it, but every other
  operator works.

### Time

- `ts` is a `datetime` (a small subclass of it, see below). It is the clock time written in the text. An offset is
  ignored, so `2020-02-25T12:00:00+00:00` and `2020-02-25T12:00:00` are the same `ts`. Sources with different
  offsets are not converted to one zone.
- Accepted text is what `datetime.fromisoformat` reads: `2026-07-25T18:15:00`, `2026-07-25 18:15:00`,
  `2026-07-25T18:15`, with `Z` or an offset, or a date alone (midnight). Other forms (epoch numbers,
  `25/07/2026 18:15`) are not read in v1: such a field stays as it came, and the row has no `ts`.
- There are no extra time fields. Everything comes from `ts` with standard `datetime` methods: `ts.hour`,
  `ts.minute`, `ts.isoweekday()` (1 = Monday), `ts.date()`.
- The subclass adds three things:
  - `ts` compares with ISO text: `"2026-07-24" <= ts < "2026-07-25"`;
  - `ts.between(a, b)`: true when `a <= ts <= b`, with both ends included, as in SQL and pandas;
  - `ts.slot(minutes=15)`: the slot of the day, `(ts.hour * 60 + ts.minute) // minutes`, so 0 to 95 for 15
    minutes. 18:00, 18:05 and 18:10 are all slot 72.
- In a reply, `ts` and every other time value (for example `ts.date()`) are written as ISO text,
  `"2026-07-25T18:15:00"` or `"2026-07-25"`.

### The interval of a row list

Every row list has an **interval**: its time step in minutes. It is called "interval" to avoid a clash with data
fields named `step` and with the chain's `steps`.

- For a source, it is measured on the rows as fetched: the smallest gap between two consecutive different `ts`.
  With fewer than two different times, the interval is unknown.
- After a `reduce` with a `window`, it is the window.
- `filter`, `map`, `top` and `collect` do not change it.
- `union` keeps the smaller interval of the two lists, and `join` keeps the interval of the incoming rows.
- After any other `reduce`, the interval is unknown.

`reduce` uses the interval to keep only complete windows (section 5).

## 5. Operators

Each operator takes a row list and returns a new one. It never changes the rows it receives.

### filter

`{"op": "filter", "where": "<expression>"}`

Keeps the rows where the expression is `True`. The expression must return `True` or `False`. Any other value is
an error (400).

### map

`{"op": "map", "set": {"<field>": "<expression>", ...}}`

Sets the fields in the order given, so a later expression can read a field set earlier. A value can be of any type:
a number, text, `True`/`False`, `None`, or a time.

### reduce

```json
{"op": "reduce", "window": "15m", "aggregate": {"kw": "mean(kw)"}}
{"op": "reduce", "by": {"slot": "ts.slot()"}, "aggregate": {"baseline_kw": "mean(kw)"}}
{"op": "reduce", "by": ["weekday", "hour"], "aggregate": {"*": "mean"}}
{"op": "reduce", "aggregate": {"charge*": "sum", "readings": "count()"}}
```

**How rows are grouped:** a reduce has at most one of `window` and `by`.

- **`window`**: `<n>m` or `<n>h` that divides a day (`15m`, `30m`, `1h`, `4h`), or `1d`, or `7d`.
  - The windows are tumbling and aligned to midnight. `7d` windows start on Monday.
  - An output row has `ts`, the start of the window, and the aggregates.
  - **Only complete windows are kept.** A window is complete when it holds (window ÷ interval) different `ts`
    values. If the interval is unknown, no window is complete.
- **`by`**: the keys of the groups. It can be written in three ways:
  - one field name: `"hour"`;
  - a list of field names: `["weekday", "hour"]`;
  - an object, output field → expression: `{"slot": "ts.slot()", "weekday": "ts.isoweekday()"}`.

  The keys are compared as they are; Java truncated them to whole numbers. An output row has the key fields and
  the aggregates.
- **Neither:** all the rows form one group, and the output is one row, as in SQL without `GROUP BY`. If there are
  no rows, the output is no rows.

**What is computed:** `aggregate` holds one or more entries. The criteria are `mean`, `sum`, `count`, `min` and
`max`.

- **Named entry:** output field → `criteria(field)`, for example `"peak_kw": "max(kw)"`. `count()` counts the rows.
- **Field entry:** field name or pattern → `criteria`, for example `"kw": "mean"` or `"charge*": "sum"`. The output
  keeps each field's name. `"*"` means every field that holds numbers in all the rows. It leaves out the keys,
  `ts`, text, `True`/`False`, and fields that hold only `null`.
- `null` values are skipped, as in SQL. `count(field)` counts the values that are not null. If a group has no
  value for a field, the aggregate is `None`.

The output is sorted by the key.

### join

`{"op": "join", "on": "slot", "with": {<chain>}}`, or `"on": ["weekday", "hour"]`

- `on` holds field names that both sides have. To join on a computed key, set it first, with `map` or with `by`.
- An inner join. A row with no match on the other side is dropped.
- The `with` chain must have one row per key. A key that occurs twice is an error (400) that names the key.
- If both sides have a field with the same name, the value from the incoming rows is kept.

### union

`{"op": "union", "with": {<chain>}}`

Appends the rows of the `with` chain after the incoming rows. Duplicate rows are kept.

### top

`{"op": "top", "n": 3, "by": "kw", "order": "desc"}`

Keeps the `n` rows with the highest value of `by`. With `"order": "asc"`, it keeps the lowest. The default is
`desc`. Rows with the same value keep their input order.

### collect

`{"op": "collect", "fields": ["ts", "kw"], "decimals": 1}`

- `fields` is optional. When given, only those fields are kept, in that order, and a field missing from a row is
  an error (400). When left out, every field is kept.
- Decimal numbers are rounded with Python's `round(v, decimals)`. The default for `decimals` is 2.
- `collect` can only be the last step, and it is optional. Without it, every field is returned and numbers are not
  rounded.

### Every reply

Times are written as ISO text. A number that is not finite (infinity, NaN) cannot be written in JSON, so it is an
error (400) that names the field.

## 6. Expressions

Expressions are used in `filter` (`where`), in `map` (`set`) and in `reduce` (`by`). Each one is a single Python
expression: it can call methods on values, but it cannot be a statement.

| Allowed | Examples |
|---|---|
| Values | `12`, `0.95`, `'text'`, `True`, `False`, `None`, field names |
| Arithmetic | `+ - * / // % **`, with Python behaviour: `-7 % 3 == 2` |
| Comparison | `< <= > >= == !=`, chained (`17 <= ts.hour < 21`), `in` / `not in` (`ts.isoweekday() in [6, 7]`, `ts.hour in range(17, 21)`), `is None`, `is not None` |
| Logic | `and`, `or`, `not` |
| A choice | `a if condition else b` |
| Indexing | `meta['origin']`, `values[0]` |
| Functions | `abs`, `min`, `max`, `round`, `sqrt`, `log`, `exp`, `floor`, `ceil`, `range`, `field('load (kW)')` |
| Methods and attributes | on times: `year`, `month`, `day`, `hour`, `minute`, `second`, `weekday()`, `isoweekday()`, `date()`, `timestamp()`, `between()`, `slot()`; on durations (`ts - other`): `days`, `seconds`, `total_seconds()`; on text: `lower()`, `upper()`, `strip()`, `startswith()`, `endswith()`; on objects: `get()`; on numbers: `is_integer()` |

- In JSON, write text inside an expression with single quotes: `"where": "meta['origin'] == 'MG2'"`.
- A name that is not a function is a field of the row. Use `field('...')` for a field whose name is not a Python
  name. Function names never clash with field names.
- **Refused when the chain is parsed (400):**
  - any attribute or method that is not in the list, including anything that starts with `_` and `format`;
  - names that start with `_`;
  - calls to any other function, including `getattr`, `eval` and `__import__`;
  - `lambda`, comprehensions, `:=`, f-strings, and statements.
- **Errors while running (400)** name the step, the expression and the reason, for example an unknown field,
  division by zero, text compared with a number, `None` in arithmetic, or `ts` missing. To avoid a division by
  zero, write `x / y if y != 0 else 0`.

How an expression is run (there is no `eval` of the text):

1. `ast.parse(text, mode="eval")`.
2. Every node of the tree is checked against the allowed lists above.
3. The field names are collected; the policy check uses them. Data can reach an expression only through a field
   name, so this set is exact: `ts.hour` reads only `ts`.
4. `**` and `*` become calls to small functions that bound their size.
5. The tree is compiled once.
6. It is evaluated for each row with no builtins: only the listed functions, and the row as the only names.

Attribute access is limited to a list because a plain `eval` with empty builtins can still be escaped through
attributes, for example `().__class__.__base__.__subclasses__()` or `'{0.__class__}'.format(x)`.

**Limits:** a maximum expression length (a setting); `**` with an exponent of at most 100; and text or lists made
with `*` of at most 10,000 items.

Java expression → expression here:

| Java | Here |
|---|---|
| `hour >= 17 && hour < 21` | `17 <= ts.hour < 21` |
| `net > 0 ? net : -net` | `abs(net)` |
| `x > 200 ? 200 : (x < -200 ? -200 : x)` | `max(-200, min(200, x))` |
| `ts >= 2026-07-25T18:15:00` | `ts >= '2026-07-25T18:15:00'` |
| `day == 59` | `'2026-07-30' <= ts < '2026-07-31'` |
| `slot` (a derived field) | `ts.slot()` |
| `loading > 1.0 ? 1 : 0`, to count the steps later | `loading > 1.0`; `sum` counts `True` as 1 |

## 7. HTTP API

| Endpoint | Body | Reply |
|---|---|---|
| `POST /chains/run` | a chain | 200: the rows |
| `POST /chains/check` | a chain | 200: `{"sources": [{"link": "...", "raw": true}, ...]}`. Only parse and policy; nothing is fetched. With the policy switched off, `raw` is `null` |
| `GET /operators` | | 200: every operator: `op`, `family` (stream or ml), `description`, and the JSON `schema` of its parameters (section 11) |
| `GET /models` | | 200: the card of every trained model; a model that can be shared has `export`, the link of its export |
| `GET /models/{name}` | | 200: one card; 404 if there is no such model |
| `GET /models/{name}/export` | | 200: `[record]`, the model as plain numbers, a source for `load` (section 11); 400 for a `gbr` model; 404 if there is no such model |
| `GET /health` | | 200 |

Errors, always in JSON:

| Status | Body | When |
|---|---|---|
| 400 | `{"error": "steps[2] map: unknown field kw2"}` | The chain cannot be read or run, or a source refuses it (section 4) |
| 403 | `{"rejected": "steps[0] filter: on raw rows a filter may only read [ts], not [kw]"}` | The chain breaks the policy |
| 502 | `{"error": "<link>: no reply after 10 s"}` | A source cannot be reached, replies 5xx, or times out |

- An error names the step by its path, for example `steps[1].with.steps[0]`, so the caller (often an LLM) can fix
  that step.
- A problem in a chain never gives 500. A 500 is a bug in the service.
- The routes are plain `def` functions: FastAPI runs them in a thread pool, and they fetch with `httpx.Client`.

## 8. Policy

**Switched off by default for now.** With `SANDBOX_POLICY_ENABLED=false` (the default), every chain is allowed and
any link is fetched; `POST /chains/check` only lists the links. The code below stays as it is, and
`SANDBOX_POLICY_ENABLED=true` switches it back on (the tests run it both ways).

`policy.json` belongs to the data owner. It is loaded at startup, and its path is a setting. The sample policy also
lists the ML operators, each with `{"onRaw": false}`: no model sees raw readings.

```json
{
  "sources": [
    {"match": "http://localhost:8000/microgrids/*/readings*", "raw": true},
    {"match": "http://localhost:8000/microgrids/*",           "raw": false}
  ],
  "timeFields": ["ts"],
  "operators": {
    "filter":  {"onRaw": "time fields only"},
    "reduce":  {"onRaw": {"minWindowMinutes": 15, "criteria": ["mean", "sum", "count"]}},
    "map":     {"onRaw": false},
    "join":    {"onRaw": false},
    "union":   {"onRaw": false},
    "top":     {"onRaw": false},
    "collect": {"onRaw": false}
  }
}
```

### Sources

- A link is allowed only if it matches a `match` pattern. The first pattern that matches decides whether the link
  is raw. A link that matches no pattern is refused (403).
- A pattern is a glob (`*`), and its scheme, host and port are written in full, with no `*` before the path. This
  stops `http://localhost:8000*` from also matching `http://localhost:8000.evil.com`.
- These links are also refused, as in the connector: links with user info, links with `..` in the path, and links
  with `%` in the path.

### Raw rows

- Rows from a raw link are raw until the first `reduce` with a `window`.
- While the rows are raw, each step must follow the `onRaw` rule of its operator:
  - `false`: the operator is refused;
  - `filter`: the expression may read only the fields in `timeFields`, so with this policy only `ts`
    (`17 <= ts.hour < 21` is allowed, `kw > 100` is not);
  - `reduce`: it must use a `window` of at least `minWindowMinutes`, and every aggregate must use one of the listed
    criteria. A reduce with `by`, or with neither, is refused.
- The output of the chain, and the output of every chain under `with` (or `from`, for `load`), must not be raw.
- An operator that the service knows but the policy does not list is refused (403), so the owner enables operators
  one at a time. An operator that the service does not know is a 400.
- The check reads only the parsed chain and the policy. It fetches nothing.

### Defaults in the sample policy

The owner can change these in the file:

- `max` and `min` are not allowed on raw rows, because a maximum returns one real reading.
- `load` is allowed on raw rows (`"onRaw": true`): it passes the rows through untouched. The link of the model it
  loads must match a source pattern like any other link; the sample policy allows `.../models/*/export` on port 8095.

### A gap in the Java runtime that this design closes

In Java, the reading interval of a window is measured *after* the filters. Take a filter on raw rows that reads
only the time, for example `ts % 15 == 0` in Java, or `ts.minute % 15 == 0` here. It keeps one 5-minute reading in
three, so the measured interval becomes 15. A `15m` window that holds that one reading then counts as complete,
and its `mean` is the raw reading. Here the interval comes from the source as fetched, so such a window is
incomplete and is dropped. This comes from reading the Java code: I have not run it, because the jar needs Java 21.

## 9. Settings

Each setting can be set in the environment as `SANDBOX_<NAME>`, or in `.env` (pydantic-settings, as in
plan-orchestrator).

| Setting | Default |
|---|---|
| `policy_enabled` | `false` |
| `policy_file` | `policy.json` |
| `model_dir` | `models` |
| `fetch_timeout_seconds` | 10 |
| `max_response_bytes` | 64 MB |
| `max_records_per_source` | 200,000 |
| `max_expression_length` | 500 characters |
| `record_time_fields` | `["ts", "timestamp", "time", "date"]` |
| `log_level` | `INFO` |

Settings say how the service runs. The policy says what a chain may do with the data.

## 10. Project layout

The conventions are those of `plan-orchestrator`: the package is `src/`, the virtualenv is `venv/`, and the tests
are pytest. Where a module replaces a Java class, that class is named on the right.

```
code-generated-on-demand-sandbox/
  DESIGN.md  README.md  policy.json  .env.example  .gitignore
  requirements.txt  requirements-dev.txt  pytest.ini  ruff.toml
  src/
    main.py                     create_app(): settings, model store, HTTP client, routers   ChainApplication
    dependencies.py             ChainServiceDep, ModelStoreDep
    exceptions.py               ChainError (400), PolicyViolation (403), SourceError (502)  PolicyViolation
    config/settings.py          Settings                                               application.properties
    config/services.py          puts the parts together; the policy check or NoPolicy
    routers/                    chains.py (/chains/run, /chains/check), operators.py, models.py, health.py,
                                errors.py (errors -> replies)                          ChainController
    services/chain_service.py   parse, check, fetch, run                               ChainService
    services/chain_parser.py    validated JSON -> Chain of operations                  ChainParser
    services/builders/          stream.py, ml.py: one builder per operator
    services/catalogue.py       the operator catalogue (GET /operators)
    models/chain_json.py        the chain JSON; models/steps/stream.py and ml.py: one model per operator
    models/model_record.py      ModelRecord: a shared model as plain numbers (export, load)
    models/chain.py             Chain: source and steps, ready to run                  Chain
    models/rows.py              Rows: the list of dicts and its interval               Row
    models/time.py              Timestamp (compare with text, between, slot); text <-> time   Time
    expressions/                expr.py, allowed.py, checker.py, rewriter.py, guards.py   Expr
    clients/                    source_client.py, replies.py, records.py               SourceClient
    policy/                     policy.py, sources.py, rules.py, policy_checker.py     AllowedSources, PolicyChecker
    operations/base.py          Operation, CombinesChains, RunContext                  Operation, CombinesChains
    operations/stream/          filter, map, reduce/ (a package), join, union, top, collect   *Operation
    operations/ml/              fit, finetune, load, predict, forecast, evaluate, embed, similarity
    ml/                         features, algorithms, trained models, metrics, profiles, vectors, sharing
    store/model_store.py        the trained models and their cards
  tests/
    conftest.py                 a local source server (like the data service), the app with the policy on or off
    utils.py                    test data, rows, running a chain without HTTP
    resources/microgrids/       hourly readings of MG1, MG2, MG3, MG11, MG12 (real data) and MG2's demand_forecast
    resources/recipes/          the chains a code generator is expected to write, per capability (section 16)
    resources/sources/, chains/ small data sets and chains of the stream tests
    scenario/                   stand-ins for the orchestrator, its knowledge base and the LLM (section 16)
    test_*.py                   one file per area; test_uc1_workflow.py and test_uc3_adaptation.py for the use cases;
                                test_model_sharing.py runs two sandboxes, each with its own model store
```

The operator contract:

```python
class Operation(ABC):
    name: ClassVar[str]                                   # as in the JSON and in policy.json

    @abstractmethod
    def apply(self, rows: Rows, ctx: RunContext) -> Rows: ...
```

`Rows` holds the list of dicts and the interval. `RunContext` holds the fetch-once cache and the model store. Each
operation is a frozen dataclass, built by its builder from plain values, so it can be tested on its own without
JSON.

**Adding an operator:**

1. Add its pydantic model in `models/steps/stream.py` or `ml.py`, and to `StepJson` in `models/chain_json.py`.
2. Write its class in `operations/stream/` or `operations/ml/`, and add it to that family's list.
3. Add a builder for it in `services/builders/stream.py` or `ml.py`.
4. Give it a rule in `policy.json`. With the policy switched on, an operator without a rule is refused.
5. Write its tests. The catalogue (`GET /operators`) lists it from then on.

**Dependencies:** fastapi, uvicorn, httpx and pydantic-settings; scikit-learn and joblib for the ML operators;
pytest and ruff for development. Rows stay lists of dicts: there is no pandas. Python 3.12.

For manual runs, any link works as a source, for example the existing data service:
`python3 edc-connectors/chain-runtime/data-service/service.py`, on port 8000.

## 11. ML operators

The ML operators train, use and compare models, in the same chain format as the stream operators. They are kept
minimal on purpose: the point is how a capability (a forecast, a model) is obtained, derived and reused across
microgrids (section 16), not the accuracy of the models.

| op | JSON | Gives |
|---|---|---|
| `fit` | `model`, `target`, `lags`, `calendar` (`hour`, `weekday`, `month`), `algo`, `params`, `holdout` | One row: the card of the trained model, which is saved in the model store |
| `finetune` | `model` (the source), `as` (the new name), `alpha`, `holdout` | One row: the card of the adapted model, saved under `as` |
| `load` | `from` (a chain that gives one model record), `as` (default: the record's name) | The rows, unchanged; the model is saved in the model store |
| `predict` | `model`, `as` | The rows, with the model's prediction; rows without every lag are dropped |
| `forecast` | `model`, `steps`, `as` | Only the next `steps` rows after the last `ts`: `{ts, <as>}` |
| `evaluate` | `actual`, `predicted` | One row: `n`, `mae`, `rmse`, `mape` (%); rows with a null are skipped |
| `embed` | `field`, `by`, `normalize`, `as` | One row per group: the hour-of-day profile (24 numbers) and `days` |
| `similarity` | `field`, `metric` (`distance`, `cosine`), `with`, `as` | The rows, with the distance (or cosine) of their vector to the one row of `with` |

### Features

A model predicts its target from **lags** and the **calendar**. `lag_k` is the target k time steps before `ts`
(k × the interval of the rows). A calendar feature is written as its sine and cosine, so 23:00 is close to 00:00.
The rows need `ts` and a known interval, so a chain on raw readings reduces them first (`"window": "1h"`). A model
only takes rows with the interval it was trained on; any other interval is a 400 that says which window to use.
`forecast` is recursive: each step uses the actual values before it, then the values it forecast.

### Algorithms

`linear` (ridge regression; `alpha` from 0 to 10⁶, default 1) and `gbr` (gradient boosting; `n_estimators` 1-1000,
`max_depth` 1-10, `learning_rate` 0.001-1). The algorithm, its parameters and their ranges are checked when the
chain is parsed, before anything runs, so a policy can later limit them in the same place. `src/ml/algorithms.py`
holds the list.

### The model store, and models as assets

`<model_dir>/<name>.joblib` holds the estimator and its features; `<name>.json` is the **card**, the model as an
asset: `kind` (`trained`, `adapted` or `loaded`), `algo`, `params`, `features`, `target`, `lags`, `calendar`,
`interval_minutes`, `rows`, `holdout_rows`, `first_ts` and `last_ts` of the training rows, `mae` and `rmse` on the
holdout rows, `source_model` (the model it was adapted or loaded from), `created_at`. `GET /models` and
`GET /models/{name}` return the cards, never the files. A model saved again under the same name replaces the old
one. Model names may only hold letters, digits and `_ . -`, because they become file names.

joblib files are loaded only from this folder, which only this sandbox writes: loading someone else's joblib file
can run code. Models move between sandboxes as plain numbers instead ("Sharing models" below). In the use-case
tests, all the microgrids share one sandbox and its store; `tests/test_model_sharing.py` runs one sandbox per node.

### Transfer: finetune

The adapted model is the source model plus a correction learned on the local rows: prediction = source(x) +
residual(x), where the residual is a ridge regression trained on (local value - source prediction). A larger
`alpha` keeps the correction smaller, so the model stays closer to the source. This is transfer learning in its
simplest form, but it has the property UC3 needs: a microgrid with a few days of data starts from the knowledge of
another one.

### Sharing models: export and load

Each node has its own sandbox and model store. A model goes from one to another by link:

1. **Export.** `GET /models/{name}/export` replies with a list of one record: the card of the model, plus `format`,
   `coef` and `intercept`. The card in `GET /models` gives this link as `export`. The record is a normal source
   reply, so the link can be an EDC asset like any data link.
2. **Load.** `{"op": "load", "from": {"source": "<export link>"}, "as": "<name>"}` fetches the record, checks it
   (`models/model_record.py`), and saves the model in the local store with `kind: "loaded"` and `source_model`. The
   incoming rows pass through unchanged, so the next steps (`predict`, `forecast`, `finetune`) can use the model in
   the same request.

```json
{"source": "http://localhost:4001/microgrids/11/readings?start=2026-07-15&end=2026-07-18",
 "steps": [
   {"op": "reduce", "window": "1h", "aggregate": {"kw": "mean"}},
   {"op": "load", "from": {"source": "http://node12:8095/models/mg12_demand/export"}, "as": "mg12_demand"},
   {"op": "finetune", "model": "mg12_demand", "as": "mg11_from_mg12", "holdout": 24}]}
```

Format `linear`, the only one for now: prediction = `intercept` + Σ `coef[i]` × feature `i`, with the features in
the order of `features` (the lags, then each calendar name as sine and cosine). The record holds names and numbers
only, so loading it cannot run code. `load` refuses a record whose format is unknown, whose `coef` does not have
one number per feature, whose lags or calendar names are not those `fit` allows, or whose numbers are not finite.

What can be shared: models trained with `linear`, and models adapted from them (source + correction is again one
linear model). Loaded models can be shared on. `gbr` models cannot yet: their export is a 400, and their card has no
`export` link.

Why the link is a chain under `from`, not a `url` parameter: a chain fetches nothing but its sources. The model link
is checked by the policy, listed by `/chains/check`, bounded by the fetch limits and named in errors
(`steps[1].from.source`), all with the code that already does this for `with`.

### Embeddings and similarity

`embed` gives the mean value of a field at each hour of the day. In kW (the default) it carries the shape and the
size of a microgrid; normalized (`"normalize": true`) only the shape. On the data set of the use cases, the 20
microgrids have almost the same daily shape (cosine > 0.99 for every pair) and differ mostly in size (150 to 530 kW
on average), so the tests compare profiles in kW with `distance`. `distance(a, b)` and `cosine(a, b)` are also
functions in expressions.

### The operator catalogue

`GET /operators` lists every operator with its family, what it does (from the docstring of its class) and the JSON
schema of its parameters (from its pydantic model). It is what a code generator works from: a generated chain may
only use the operators the catalogue lists (section 16).

### Examples

A training pipeline (the chain a generator writes for a microgrid without a demand model):

```json
{"source": "http://localhost:8000/microgrids/1/readings",
 "steps": [
   {"op": "reduce", "window": "1h", "aggregate": {"kw": "mean"}},
   {"op": "fit", "model": "mg1_demand_model", "target": "kw", "lags": [1, 2, 24], "calendar": ["hour", "weekday"],
    "holdout": 24}]}
```

The load of the next 4 hours, with that model:

```json
{"source": "http://localhost:8000/microgrids/1/readings",
 "steps": [
   {"op": "reduce", "window": "1h", "aggregate": {"kw": "mean"}},
   {"op": "forecast", "model": "mg1_demand_model", "steps": 4, "as": "load"}]}
```

MG1's model adapted to 4 days of MG11, and validated on the last of them:

```json
{"source": "http://localhost:8000/microgrids/11/readings?start=2026-07-15&end=2026-07-18",
 "steps": [
   {"op": "reduce", "window": "1h", "aggregate": {"kw": "mean"}},
   {"op": "finetune", "model": "mg1_demand_model", "as": "mg11_from_mg1", "holdout": 24}]}
```

### Not yet

Jobs for long runs (`POST /jobs`); sharing `gbr` models (their trees as JSON, or ONNX); forecasting several steps at once (direct
multi-output); a `featurize` operator to see the features (they are built inside the operators, from the card, so
they are always the same for training and use).

## 12. Changes from the Java runtime

| Java | Here | Why |
|---|---|---|
| `"source": "MG_n"` with `start`/`end` | links only | One way to name a source |
| The chain says `"raw": true` | `policy.json` says which links are raw | A caller cannot leave out `raw` to skip the policy |
| Any link is fetched | Only links that match a policy pattern; no redirects | The service cannot be sent to any address |
| Its own language: `&&`, `?:`, times without quotes | Single Python expressions, with methods on values | A known syntax; no parser to write; `ts.hour`, `abs()`, `max()` |
| Derived fields `day`, `weekday`, `hour`, `slot` | `ts.date()`, `ts.isoweekday()`, `ts.hour`, `ts.slot()` | One way to read the time; not tied to one data set (`day` counted from 2026-06-01) |
| `%` takes the sign of the left side; x/0 gives Infinity; rounding half up | Python behaviour; x/0 is an error | No hidden Infinity in results: Java wrote Infinity as 9223372036854775807, and NaN as 0 |
| Only numbers are kept from records | Every JSON value is kept | Text and flags can be used; vectors later |
| A true/false field set by `map` cannot be read later | It can be read | |
| `reduce`: one field, and a key is required | `aggregate` with several fields, patterns and `*`; `by` with expressions; no key gives one row | One reduce in place of a branch and a join for each value |
| Keys truncated to whole numbers | Keys as they are | A join on 0.5 does not merge with 0 |
| The window interval is measured after filters | The interval comes from the source | Closes the gap described in section 8 |
| An unknown operator is 403 | 400; an operator missing from the policy is 403 | A typo is not a policy matter |
| Several malformed chains give 500 | 400, or 502 for sources | |
| `max` on raw rows: `policy.json` allows it, the docs forbid it | Not allowed by default | One rule, in the policy file |
| `top`: highest values only | `order`: `asc` or `desc` | No `-x` trick to get the lowest values |
| `collect` needs the list of fields, and is needed to write `ts` as text | `fields` is optional; times are always written as text | |

## 13. Worked examples

Both use the MG2 files of the data service. They become end-to-end checks in task 10.

**A. Sum the battery flexibility from 13:00 to 14:00 into one row** (MG2 `battery_flex`, not raw):

```json
{"source": "http://localhost:8000/microgrids/2/battery_flex",
 "steps": [
   {"op": "filter", "where": "ts.between('2020-02-25T13:00:00+00:00', '2020-02-25T14:00:00+00:00')"},
   {"op": "reduce", "aggregate": {"charge*": "sum", "discharge*": "sum"}},
   {"op": "collect", "decimals": 4}]}
```

```json
[{"charge_p10": 1179.8444, "charge": 1201.2534, "charge_p90": 1235.0528,
  "discharge_p10": -49.9416, "discharge": -16.7093, "discharge_p90": 132.3328}]
```

With `{"*": "sum"}`, the field `step` is summed too (26).

**B. Flexibility: the 15-minute forecast minus the usual load of that slot** (MG2 `demand_forecast`, not raw,
and 4 weeks of MG2 readings, raw):

```json
{"source": "http://localhost:8000/microgrids/2/demand_forecast",
 "steps": [
   {"op": "map", "set": {"slot": "ts.slot()"}},
   {"op": "join", "on": "slot", "with": {
     "source": "http://localhost:8000/microgrids/2/readings?start=2026-07-03&end=2026-07-30",
     "steps": [
       {"op": "reduce", "window": "15m", "aggregate": {"kw": "mean(kw)"}},
       {"op": "reduce", "by": {"slot": "ts.slot()"}, "aggregate": {"baseline_kw": "mean(kw)"}}]}},
   {"op": "map", "set": {"diff_kw": "load - baseline_kw", "flex_kw": "max(0, load - baseline_kw)"}},
   {"op": "collect", "fields": ["ts", "slot", "load", "baseline_kw", "diff_kw", "flex_kw"], "decimals": 1}]}
```

- The readings come every 5 minutes. The `15m` window turns them into one row per 15 minutes, and its start is the
  slot, so 18:00, 18:05 and 18:10 all count towards slot 72. The policy requires this window on raw rows anyway.
- The two sides join on the time of day (`slot`), because the forecast is dated 2020 and the readings 2026.

10 rows come back. Slot 51 (12:45) is the only one with a surplus: load 428.0, baseline 421.2, flex 6.8. Slots 72
and 73 have baselines of 645.1 and 644.6 kW. Each is the mean of the 84 raw readings in that slot (3 per day, 28
days), checked against the CSV.

Without the `15m` window, the check refuses the chain before anything is fetched:
`steps[1].with.steps[0] reduce: on raw rows the first reduce must use a window`.

**Open point for B:** this assumes that a forecast's `timestamp` is the start of its 15 minutes, as for the
`reduce` windows. In the file, step 1 has `timestamp` equal to `origin`, which suggests start labels. If the
forecast labels the end of each period, use `(ts.slot() - 1) % 96` on the forecast side.

## 14. Process and tasks

**How each task is done:**

1. I build one task, with its tests.
2. I run `pytest` and `ruff`.
3. I explain what was added, how it connects to the earlier tasks, and how to try it.
4. I stop. The next task starts when you say so.

If something in a task shows that this design is wrong, I stop and propose the change here first.

| # | Task | Done when |
|---|---|---|
| 1 | **Skeleton**: requirements, `pytest.ini`, `.env.example`, `.gitignore`, `config/settings.py`, `exceptions.py`, `main.py` with `create_app()`, `GET /health` | `pytest` passes; `uvicorn src.main:app` answers `/health` |
| 2 | **Time**: `models/time.py`: `Timestamp` (compare with text, `between`, `slot`), read text → `Timestamp` with the offset dropped, `Timestamp` → text | Tests for every accepted form, offsets, comparisons with text, `between` ends, `slot` at the slot edges |
| 3 | **Expressions**: `expressions/expr.py` and `allowed.py`: parse, check the nodes, collect fields, bound `**` and `*`, compile, evaluate; errors with a reason | Tests for each allowed form, each refused form (including the escape examples), `fields()`, and the run-time errors |
| 4 | **Chain format**: `models/chain_json.py` (the 7 operators), `models/chain.py`, `services/chain_parser.py` with the operator cases, `operations/base.py` | Valid chains parse; unknown keys, wrong types and bad expressions give one-line errors with the step path |
| 5 | **Sources**: `clients/source_client.py`: GET, the three reply forms, records → rows, `ts`, the interval, the limits, 400 vs 502; the local test server in `tests/conftest.py` | Tests against the test server for each reply form, a 4xx, a 5xx, a timeout, too many records, and the interval |
| 6 | **filter, map, top, collect**, `models/rows.py`, and running a chain without `with` in `services/chain_service.py` | Tests for each operator, including `collect` without `fields` and a non-finite number |
| 7 | **reduce**: `window` with complete windows, `by` in its three forms, no key, both kinds of `aggregate` entry, `*`, nulls, sorting | Tests for each case; example A gives its row from the test data |
| 8 | **join and union**: chains under `with`, `RunContext` with each link fetched once | Tests for matches, a duplicate key, name clashes, union order, one fetch per link; example B runs on the test data |
| 9 | **Policy**: `policy.json`, `policy/policy.py` (patterns, link checks), `policy/policy_checker.py` | Tests for every refusal in section 8, including the thinning case and `with` chains that stay raw |
| 10 | **HTTP**: `routers/chains.py` (`/chains/check`, `/chains/run`), the error replies, `dependencies.py`; README with the examples | End-to-end tests through HTTP for 200, 400, 403 and 502; examples A and B against the data service give the rows in section 13 |

After task 10: the ML design (section 11), to approve, and then its tasks.

## 15. Decisions

Agreed in the discussion of 2026-11-20:

1. A new project with no compatibility target. Sources are links. The connector comes later.
2. Chains are JSON with a fixed operator list, run by an interpreter. There is no generated Python.
3. Synchronous for now. Async jobs stay open.
4. The layout follows plan-orchestrator: FastAPI, `src/`, `venv/`.
5. Expressions are single Python expressions with methods on values (option A). There are no derived time fields;
   `by` takes expressions.
6. `ts` adds `between()` (both ends included) and `slot()` to `datetime`, and compares with ISO text.
7. A `reduce` with no key gives one row; `aggregate` has named entries, field entries, patterns and `*`.
8. `collect` `fields` is optional. The time step of a row list is called the "interval".
9. `ts` is read from `ts`, `timestamp`, `time` or `date` when it holds ISO text.

To confirm with this draft:

10. The replies are 400, 403 and 502. An unknown operator is 400.
11. The sample policy allows no `max`/`min` on raw rows.
12. Aggregates skip nulls, as in SQL.
13. Source patterns are globs, each with `raw`; the first match wins.
14. The project folder is `tmp/code-generated-on-demand-sandbox/`, where this file is. It can be moved later.

Open: whether forecast timestamps mark the start or the end of their period (section 13, example B).

Added for v2 (2026-11-20):

15. The policy check is switched off by default; the code stays and a setting switches it back on.
16. Operators are in two families, `operations/stream/` and `operations/ml/`.
17. The knowledge base, the orchestration and the LLM are not part of the sandbox: the tests have stand-ins for
    them (section 16). The sandbox offers what they need: operators, a catalogue, a model store, and checks.

Added for v2.1 (2026-11-23):

18. Models move between sandboxes as a record of plain numbers (`GET /models/{name}/export`), never as model files;
    linear models first.
19. `load` takes the model link as a chain under `from`, not as a `url` parameter, so a chain fetches nothing but its
    sources. It passes the rows through, so a model can be loaded and used in one request.

## 16. The use cases in the tests: code generated on demand

The novelty the tests show is in the data management, not in the models: when a step of a workflow needs a
capability that a microgrid does not have, the system **derives it with generated code** (a chain of the
sandbox's operators) instead of breaking the workflow, checks that code before it registers it, and **reuses** it
for the next microgrids. In UC3 it also **adapts** each node: it picks, per microgrid, the best way to obtain a
model among the alternatives that the other microgrids make possible.

The orchestrator, its knowledge base and the LLM are not part of the sandbox. `tests/scenario/` has small
stand-ins for them, so the tests can drive the sandbox over HTTP as the system will.

### The knowledge base and the code-generation logic

- **Asset**: what a microgrid offers for a capability: data (a link, `agg`), predictions (a link, `pred`) or a
  model (a model name, `model`). Example: MG2 offers `demand_forecast` as a file; MG1 offers only `load_readings`.
- **Recipe**: how a capability is derived: the capabilities it `needs`, a chain template with `${need}`
  placeholders, what it `produces` (rows, or a model) and the `output_fields` it promises.
  `tests/resources/recipes/` holds the chains an LLM is expected to write:

  | Capability | Needs | Chain |
  |---|---|---|
  | `demand_model` | `load_readings` | reduce 1h, `fit` (the training pipeline) |
  | `demand_forecast` | `load_readings`, `demand_model` | reduce 1h, `forecast` 4 steps |
  | `consumer_flexibility` | `demand_forecast`, `load_readings` | the forecast chain, then the hourly baseline joined on the hour, `max(0, load - baseline)` |

  `{"extend": "${demand_forecast}", "steps": [...]}` builds a capability on another: whatever chain gives the
  forecast of a microgrid (its own file, or a derived forecast) is extended with the new steps.

For each microgrid of a step (`tests/scenario/orchestrator.py`):

1. an asset of the microgrid with the capability is used as it is;
2. else the recipe of the knowledge base derives it, after obtaining what the recipe needs, in the same way;
3. else the **generator** (the LLM) is asked for a recipe, with the capability, the microgrid, the assets and the
   operator catalogue. The generated code is then checked three times before it is registered:
   - it may only use operators that the catalogue lists;
   - the sandbox must accept the chain (`POST /chains/check`: shape, expressions, algorithm parameters, policy);
   - its output must have the fields the capability promises.

   Only then is the recipe registered; the next microgrid that lacks the capability reuses it without asking the
   generator. A recipe that trains a model adds the model as an asset of that microgrid, so later steps use it
   instead of training again.

### UC1 (`tests/test_uc1_workflow.py`)

A workflow of two steps over 1, 2 and 3 microgrids: the demand forecast of the next 4 hours, then the consumer
flexibility built on it. MG2 has its own `demand_forecast` asset; MG1 and MG3 have only their readings.

| Scope | demand_forecast | Generator requests |
|---|---|---|
| MG1 | derived: a model is trained, then the forecast | `demand_forecast`, `demand_model` (for MG1) |
| MG1, MG2 | MG1 derived; MG2's own file | the same two |
| MG1, MG2, MG3 | MG1 and MG3 derived; MG2's own file | the same two: MG3 reuses the registered recipes |

The flexibility step adds one request (`consumer_flexibility`) for the whole scope, and trains nothing: it reuses
the models of the first step. Three more tests give the generator broken code: an operator the catalogue does not
list, a chain the sandbox cannot read, an output without a promised field. Each one is stopped, and nothing is
registered.

### UC3 (`tests/test_uc3_adaptation.py`)

A microgrid with 4 days of history needs a demand model (`tests/scenario/adaptation.py`):

1. discovery: the demand models of the other microgrids;
2. suitability: their microgrids ranked by the distance of their profile to the target's (`embed`, `similarity`);
3. candidates: train on its own days; and for the closest source (or the `max_sources` closest), reuse its model
   as it is, or adapt it (`finetune`);
4. validation: every candidate is scored on the last day of the target's history;
5. the best becomes the target's `demand_model` asset, with where it came from; the UC1 steps then use it.

| Scope | Ranking | Chosen | MAE on the next 2 days: chosen / local only |
|---|---|---|---|
| MG3 alone | none | trained on its own 4 days: the workflow does not break | 36.9 kW (no other choice) |
| MG3 with MG1 | MG1 | MG1's model adapted to MG3 | 20.7 / 36.9 kW |
| MG11 with MG1 and MG12 | MG12 (distance 148), MG1 (585) | MG12's model as it is | 11.8 / 43.3 kW |

What the tests also show: with `max_sources` = 2 (both sources tried: 5 candidates instead of 3), the choice is
still about twice as good as training locally, but one validation day favours MG1's adapted model (22.2 kW on the
next days), where MG12's model as it is (11.8 kW) was the better choice. Ranking by similarity first is cheaper
and, here, also better; a longer validation would make the choice more reliable.
