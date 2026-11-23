"""Makes tests/resources/microgrids/MG<n>/readings.json for the microgrids the tests use.

The values are the hourly means of the 5-minute readings that the data service serves
(edc-connectors/chain-runtime/data-service/data/MG<n>/MG_<n>.csv: the 20 microgrids of the use cases, from
https://data.mendeley.com/datasets/s7dg7wfzsz/2). MG1-MG10 are in Cluj-Napoca, MG11-MG20 in Athens. The CSV column
ts counts minutes from 2026-06-01 00:00, a Monday, and the data covers 60 days.

Run it again only to change the data:
    python tests/resources/microgrids/make_readings.py ../../edc-connectors/chain-runtime/data-service/data
"""

import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

MICROGRIDS = (1, 2, 3, 11, 12)
DAY_ZERO = datetime(2026, 6, 1)


def hourly_readings(csv_file: Path) -> list[dict[str, object]]:
    by_hour: dict[int, list[float]] = {}
    with csv_file.open() as lines:
        for record in csv.DictReader(lines):
            by_hour.setdefault(int(record["ts"]) // 60, []).append(float(record["kw"]))
    return [
        {"timestamp": (DAY_ZERO + timedelta(hours=hour)).isoformat(), "kw": round(sum(values) / len(values), 3)}
        for hour, values in sorted(by_hour.items())
    ]


def main(data_folder: Path) -> None:
    here = Path(__file__).parent
    for number in MICROGRIDS:
        readings = hourly_readings(data_folder / f"MG{number}" / f"MG_{number}.csv")
        target = here / f"MG{number}" / "readings.json"
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps({"microgrid": number, "readings": readings}, separators=(",", ":")))
        print(f"{target}: {len(readings)} hourly readings")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
