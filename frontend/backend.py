import os
import sys
import time
import json
import uuid
import subprocess
from pathlib import Path
from threading import Thread, Lock
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

ROOT = Path(__file__).parent.parent.resolve()
STRATEGIES_DIR = ROOT / "strategies"

_current_process = None
_current_run_id = None
_runs: Dict[str, Dict] = {}
_lock = Lock()


def list_strategies():
    out = []
    if not STRATEGIES_DIR.exists():
        return out
    for p in STRATEGIES_DIR.iterdir():
        if p.is_dir():
            cfg = p / 'config.json'
            if cfg.exists():
                out.append(p.name)
    return out


@app.get('/strategies')
def get_strategies():
    return JSONResponse(content={'strategies': list_strategies()})


@app.get('/strategy/{name}/config')
def get_strategy_config(name: str):
    cfg_path = STRATEGIES_DIR / name / 'config.json'
    if not cfg_path.exists():
        raise HTTPException(status_code=404, detail='Strategy not found')
    with open(cfg_path, 'r') as f:
        return JSONResponse(content=json.load(f))


@app.put('/strategy/{name}/config')
def update_strategy_config(name: str, config: Dict):
    cfg_path = STRATEGIES_DIR / name / 'config.json'
    if not cfg_path.exists():
        raise HTTPException(status_code=404, detail='Strategy not found')
    with open(cfg_path, 'w') as f:
        json.dump(config, f, indent=4)
    return JSONResponse(content={'status': 'ok'})


def _monitor_process(proc: subprocess.Popen, run_id: str):
    
    global _current_process, _current_run_id
    try:
        _runs[run_id]['status'] = 'running'
        proc.wait()
    finally:
        with _lock:
            _runs[run_id]['status'] = 'finished'
           
            if _current_process is proc:
                _current_process = None
                _current_run_id = None


def _find_latest_report_dir(strategy_name: str, timeout: float = 30.0) -> Optional[Path]:
    base = STRATEGIES_DIR / strategy_name / 'report'
    start = time.time()
    while time.time() - start < timeout:
        if base.exists() and base.is_dir():
            subs = [d for d in base.iterdir() if d.is_dir()]
            if subs:
                subs_sorted = sorted(subs, key=lambda d: d.stat().st_ctime, reverse=True)
                return subs_sorted[0]
        time.sleep(0.5)
    return None

@app.post('/start')
def start_backtest(payload: Dict):
    
    global _current_process, _current_run_id
    strategy = payload.get('strategy')
    if not strategy:
        raise HTTPException(status_code=400, detail='Missing strategy')
    cfg = payload.get('config')
    allow_override = payload.get('allow_override', False)
    start_date_str = payload.get('start_date', '2025-01-01')
    end_date_str = payload.get('end_date', '2025-01-31')

    if strategy not in list_strategies():
        raise HTTPException(status_code=404, detail='Strategy not found')

    with _lock:
        if _current_process is not None:
            if not allow_override:
                raise HTTPException(status_code=409, detail='A backtest is already running')
            else:
               
                try:
                    _current_process.terminate()
                except Exception:
                    pass

    cfg_path = STRATEGIES_DIR / strategy / 'config.json'
    if cfg is not None:
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f, indent=4)

    run_id = str(uuid.uuid4())
    run_info = {
        'strategy': strategy,
        'pid': None,
        'status': 'starting',
        'run_dir': None,
    }
    _runs[run_id] = run_info

    main_py = ROOT / 'main.py'
    if not main_py.exists():
        raise HTTPException(status_code=500, detail='main.py not found')

    cmd = [sys.executable, str(main_py)]

    env = os.environ.copy()
    env['BACKTEST_START_DATE'] = start_date_str
    env['BACKTEST_END_DATE'] = end_date_str
    env['BACKTEST_STRATEGY'] = strategy

    try:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to start backtest: {e}')

    with _lock:
        _current_process = proc
        _current_run_id = run_id
        _runs[run_id]['pid'] = proc.pid
        _runs[run_id]['status'] = 'running'

    t = Thread(target=_monitor_process, args=(proc, run_id), daemon=True)
    t.start()

    report_dir = _find_latest_report_dir(strategy, timeout=20.0)
    if report_dir:
        _runs[run_id]['run_dir'] = str(report_dir)
    else:
        _runs[run_id]['run_dir'] = None

    Thread(target=_capture_process_output, args=(proc, run_id, strategy), daemon=True).start()

    return JSONResponse(content={'run_id': run_id})


def _capture_process_output(proc: subprocess.Popen, run_id: str, strategy: str):
   
    buffer_lines = []
    while True:
        line = proc.stdout.readline()
        if line == '' and proc.poll() is not None:
            break
        if line:
            buffer_lines.append(line)
           
            run_dir = _runs.get(run_id, {}).get('run_dir')
            if not run_dir:
                rd = _find_latest_report_dir(strategy, timeout=0.1)
                if rd:
                    _runs[run_id]['run_dir'] = str(rd)
                    run_dir = _runs[run_id]['run_dir']
            if run_dir:
                try:
                    log_path = Path(run_dir) / 'backtest.log'
                    with open(log_path, 'a', encoding='utf-8') as f:
                        
                        f.writelines(buffer_lines)
                        buffer_lines = []
                except Exception:
                    pass
    run_dir = _runs.get(run_id, {}).get('run_dir')
    if run_dir and buffer_lines:
        try:
            log_path = Path(run_dir) / 'backtest.log'
            with open(log_path, 'a', encoding='utf-8') as f:
                f.writelines(buffer_lines)
        except Exception:
            pass


@app.get('/runs/{run_id}/status')
def run_status(run_id: str):
    r = _runs.get(run_id)
    if not r:
        raise HTTPException(status_code=404, detail='Run not found')
    return JSONResponse(content={'status': r['status'], 'run_dir': r.get('run_dir')})


@app.get('/runs/{run_id}/log')
def tail_log(run_id: str, offset: int = 0):
    r = _runs.get(run_id)
    if not r:
        raise HTTPException(status_code=404, detail='Run not found')
    run_dir = r.get('run_dir')
    if not run_dir:
        rd = _find_latest_report_dir(r['strategy'], timeout=0.5)
        if rd:
            r['run_dir'] = str(rd)
            run_dir = r['run_dir']
        else:
            return JSONResponse(content={'lines': [], 'offset': 0})

    log_path = Path(run_dir) / 'backtest.log'
    if not log_path.exists():
        return JSONResponse(content={'lines': [], 'offset': 0})

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            data = f.read()
            new_offset = f.tell()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to read log: {e}')

    lines = data.splitlines()
    return JSONResponse(content={'lines': lines, 'offset': new_offset})


@app.get('/runs')
def list_runs():
    return JSONResponse(content={'runs': _runs})


@app.post('/runs/{run_id}/stop')
def stop_backtest(run_id: str):

    global _current_process, _current_run_id
    r = _runs.get(run_id)
    if not r:
        raise HTTPException(status_code=404, detail='Run not found')
    
    if run_id == _current_run_id and _current_process is not None:
        try:
            _current_process.terminate()
            time.sleep(0.5)
            if _current_process.poll() is None: 
                _current_process.kill()
            _runs[run_id]['status'] = 'stopped'
            return JSONResponse(content={'status': 'stopped'})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Failed to stop process: {e}')
    else:
        return JSONResponse(content={'status': 'not_running'})
