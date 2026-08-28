"""Deterministic local exports for the append-only metric ledger."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from src.runtime.metrics import MetricLedger


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_summary(ledger: MetricLedger) -> dict[str, Any]:
    events = ledger.read_events()
    series: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for event in events:
        key = (
            event.run_id,
            event.attempt_id,
            event.stage_name,
            event.split,
            event.metric_name,
            event.namespace,
        )
        series.setdefault(key, []).append({
            "created_at": event.created_at,
            "epoch": event.epoch,
            "global_step": event.global_step,
            "sample_count": event.sample_count,
            "value": event.value,
        })
    return {
        "schema_version": 1,
        "event_count": len(events),
        "run_ids": sorted({event.run_id for event in events}),
        "attempt_ids": sorted({event.attempt_id for event in events}),
        "namespace_counts": {
            namespace: sum(event.namespace == namespace for event in events)
            for namespace in ("exploratory", "selection", "final")
        },
        "series": [
            {
                "run_id": key[0],
                "attempt_id": key[1],
                "stage_name": key[2],
                "split": key[3],
                "metric_name": key[4],
                "namespace": key[5],
                "points": sorted(
                    points,
                    key=lambda point: (
                        point["epoch"], point["global_step"], point["created_at"],
                    ),
                ),
            }
            for key, points in sorted(series.items())
        ],
    }


def _dashboard_html(summary: dict[str, Any]) -> str:
    embedded = _canonical_json(summary).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GEODE Metric History</title>
<style>
:root {{ color-scheme: light; --ink:#18201c; --muted:#5d6862; --line:#cbd5cf; --paper:#f4f7f5; --panel:#ffffff; --accent:#136f63; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:linear-gradient(135deg,#edf4ef,#f8f4ea); color:var(--ink); font-family:Georgia,"Times New Roman",serif; letter-spacing:0; }}
main {{ width:min(1120px,calc(100% - 32px)); margin:32px auto; }}
header {{ display:flex; justify-content:space-between; align-items:end; gap:24px; border-bottom:2px solid var(--ink); padding-bottom:16px; }}
h1 {{ margin:0; font-size:clamp(28px,5vw,52px); font-weight:500; }}
.count {{ color:var(--muted); font:14px Consolas,monospace; }}
.toolbar {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:24px 0; }}
label {{ display:grid; gap:6px; color:var(--muted); font:13px Consolas,monospace; }}
select {{ width:100%; border:1px solid var(--line); background:var(--panel); padding:10px; font:14px Consolas,monospace; }}
.plot {{ min-height:320px; border:1px solid var(--line); background:var(--panel); padding:20px; }}
canvas {{ width:100%; height:280px; display:block; }}
table {{ width:100%; margin-top:20px; border-collapse:collapse; background:var(--panel); font:13px Consolas,monospace; }}
th,td {{ padding:10px; border-bottom:1px solid var(--line); text-align:left; }}
th {{ color:var(--muted); font-weight:400; }}
@media (max-width:700px) {{ header {{ align-items:start; flex-direction:column; }} .toolbar {{ grid-template-columns:1fr; }} main {{ margin:20px auto; }} }}
</style>
</head>
<body>
<main>
<header><h1>Metric History</h1><div class="count" id="count"></div></header>
<section class="toolbar">
<label>Metric<select id="metric"></select></label>
<label>Split<select id="split"></select></label>
<label>Namespace<select id="namespace"></select></label>
</section>
<section class="plot"><canvas id="chart"></canvas></section>
<table><thead><tr><th>Attempt</th><th>Stage</th><th>Epoch</th><th>Step</th><th>Value</th></tr></thead><tbody id="rows"></tbody></table>
</main>
<script>
const report={embedded};
const selectors={{metric:document.querySelector('#metric'),split:document.querySelector('#split'),namespace:document.querySelector('#namespace')}};
function values(field){{return [...new Set(report.series.map(item=>item[field]))].sort();}}
function fill(name,field){{for(const value of values(field)){{const option=document.createElement('option');option.value=value;option.textContent=value;selectors[name].append(option);}}}}
fill('metric','metric_name');fill('split','split');fill('namespace','namespace');
function draw(){{
 const chosen=report.series.filter(item=>item.metric_name===selectors.metric.value&&item.split===selectors.split.value&&item.namespace===selectors.namespace.value);
 const points=chosen.flatMap(item=>item.points.map(point=>({{...point,attempt:item.attempt_id,stage:item.stage_name}})));
 document.querySelector('#count').textContent=`${{report.event_count}} events · ${{points.length}} visible`;
 const canvas=document.querySelector('#chart');const scale=window.devicePixelRatio||1;canvas.width=canvas.clientWidth*scale;canvas.height=280*scale;
 const context=canvas.getContext('2d');context.scale(scale,scale);const width=canvas.clientWidth,height=280;context.clearRect(0,0,width,height);
 if(points.length){{const vals=points.map(point=>point.value),min=Math.min(...vals),max=Math.max(...vals),span=max-min||1;context.strokeStyle='#136f63';context.lineWidth=2;context.beginPath();points.sort((a,b)=>a.global_step-b.global_step).forEach((point,index)=>{{const x=24+(width-48)*(index/Math.max(points.length-1,1));const y=20+(height-40)*(1-(point.value-min)/span);index?context.lineTo(x,y):context.moveTo(x,y);}});context.stroke();}}
 const body=document.querySelector('#rows');body.replaceChildren();for(const point of points){{const row=document.createElement('tr');for(const value of [point.attempt,point.stage,point.epoch,point.global_step,point.value]){{const cell=document.createElement('td');cell.textContent=value;row.append(cell);}}body.append(row);}}
}}
Object.values(selectors).forEach(selector=>selector.addEventListener('change',draw));draw();
</script>
</body>
</html>
"""


def export_metric_history(
    ledger: MetricLedger,
    output_directory: str | Path,
) -> dict[str, str]:
    """Write deterministic JSON, CSV, and standalone HTML history exports."""
    output = Path(output_directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"{output.name}.partial-", dir=output.parent))
    try:
        summary = _build_summary(ledger)
        (temporary / "history.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        with (temporary / "events.csv").open("w", encoding="utf-8", newline="") as stream:
            fieldnames = list(ledger.read_events()[0].to_dict()) if summary["event_count"] else [
                "schema_version", "event_id", "run_id", "attempt_id", "stage_name",
                "split", "metric_name", "value", "sample_count", "created_at",
                "epoch", "global_step", "namespace",
            ]
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(event.to_dict() for event in ledger.read_events())
            stream.flush()
            os.fsync(stream.fileno())
        (temporary / "dashboard.html").write_text(
            _dashboard_html(summary), encoding="utf-8", newline="\n",
        )
        if output.exists():
            expected = {path.name: path.read_bytes() for path in temporary.iterdir()}
            actual = {path.name: path.read_bytes() for path in output.iterdir()}
            if actual != expected:
                raise ValueError("history export already exists with different content")
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, output)
        return {
            path.name: _file_hash(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        }
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise