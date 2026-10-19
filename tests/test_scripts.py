"""Tests for deployment runbook and automation scripts (T-20)."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from xau_bot.cli.main import build_parser
from xau_bot.config import Settings
from xau_bot.server import HealthCheckHandler


def get_available_powershell_runtimes() -> list[str]:
    runtimes: list[str] = []
    pwsh = shutil.which("pwsh")
    if pwsh:
        runtimes.append(pwsh)
    ps = shutil.which("powershell")
    if ps and ps not in runtimes:
        runtimes.append(ps)
    return runtimes


AVAILABLE_POWERSHELL_RUNTIMES = get_available_powershell_runtimes()


def has_dry_run_nssm_call(stdout: str, args: str) -> bool:
    """When NSSM is installed the script prints its full path instead of bare nssm.exe."""
    pattern = rf"\[DRY-RUN\] (?:\S*[\\/])?nssm\.exe {re.escape(args)}"
    return re.search(pattern, stdout, flags=re.IGNORECASE) is not None


@pytest.fixture
def scripts_dir() -> Path:
    return Path(__file__).parent.parent / "scripts"


@pytest.fixture
def docs_dir() -> Path:
    return Path(__file__).parent.parent / "docs"


assert AVAILABLE_POWERSHELL_RUNTIMES, (
    "At least one PowerShell runtime (pwsh or powershell) must be installed on Windows."
)


@pytest.fixture(params=AVAILABLE_POWERSHELL_RUNTIMES)
def powershell_exe(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def test_powershell_scripts_exist_and_have_content(scripts_dir: Path) -> None:
    verify_script = scripts_dir / "verify-vps.ps1"
    service_script = scripts_dir / "service-nssm.ps1"

    assert verify_script.exists(), "verify-vps.ps1 must exist"
    assert service_script.exists(), "service-nssm.ps1 must exist"
    assert verify_script.stat().st_size > 500
    assert service_script.stat().st_size > 500


def test_verify_vps_execution_json(scripts_dir: Path, powershell_exe: str, tmp_path: Path) -> None:
    verify_script = scripts_dir / "verify-vps.ps1"
    guardian_dir = tmp_path / "mt5_common_files"
    guardian_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["XAU_GUARDIAN_COMMON_DIR"] = str(guardian_dir)
    env["XAU_MT5_SERVER"] = "ICMarketsSC-Demo"
    env["XAU_MT5_LOGIN"] = "98765432"
    env["XAU_MT5_PASSWORD"] = "secret123"

    cmd = [
        powershell_exe,
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(verify_script),
        "-Json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)

    # In local testing with guardian_dir provided, exit code should be 0
    # (or 1 only if critical host-level check fails)
    assert proc.returncode in (0, 1)

    data = json.loads(proc.stdout)
    assert "Timestamp" in data
    assert "Summary" in data
    assert "Checks" in data

    summary = data["Summary"]
    assert isinstance(summary["Pass"], int)
    assert summary["Pass"] > 0

    checks = data["Checks"]
    assert isinstance(checks, list)
    categories = {c["Category"] for c in checks}
    assert "OS" in categories
    assert "Runtime" in categories
    assert "Clock" in categories
    assert "Guardian" in categories
    assert "Config" in categories


def test_verify_vps_masks_login_and_requires_guardian_dir(
    scripts_dir: Path, powershell_exe: str, tmp_path: Path
) -> None:
    verify_script = scripts_dir / "verify-vps.ps1"

    # 1. Test when XAU_GUARDIAN_COMMON_DIR is missing -> should report FAIL for CommonFilesDir
    env_missing_dir = os.environ.copy()
    env_missing_dir.pop("XAU_GUARDIAN_COMMON_DIR", None)
    cmd = [
        powershell_exe,
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(verify_script),
        "-Json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env_missing_dir)
    assert proc.returncode == 1, "Script must exit with 1 when critical checks fail"
    data = json.loads(proc.stdout)
    assert data["Summary"]["Fail"] >= 1
    checks_dict = {c["Check"]: c for c in data["Checks"]}
    assert checks_dict["CommonFilesDir"]["Status"] == "FAIL"

    # 2. Test when XAU_MT5_LOGIN is provided -> must be masked, plain text must not leak
    plain_login = "12345678"
    guardian_dir = tmp_path / "valid_guardian_dir"
    guardian_dir.mkdir(parents=True, exist_ok=True)

    env_with_secrets = os.environ.copy()
    env_with_secrets["XAU_GUARDIAN_COMMON_DIR"] = str(guardian_dir)
    env_with_secrets["XAU_MT5_SERVER"] = "DemoServer"
    env_with_secrets["XAU_MT5_LOGIN"] = plain_login
    env_with_secrets["XAU_MT5_PASSWORD"] = "SuperSecretPassword123"

    proc2 = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env_with_secrets)
    data2 = json.loads(proc2.stdout)
    checks2_dict = {c["Check"]: c for c in data2["Checks"]}

    assert checks2_dict["MT5Credentials"]["Status"] == "PASS"
    details = checks2_dict["MT5Credentials"]["Details"]
    assert plain_login not in details, "Plain text MT5 login must NOT leak in verification report"
    assert "******78" in details, "Login must be masked showing only last two digits"
    assert "SuperSecretPassword123" not in proc2.stdout, "Password must never appear in report"


def test_service_nssm_dry_run_install(scripts_dir: Path, powershell_exe: str) -> None:
    service_script = scripts_dir / "service-nssm.ps1"
    cmd = [
        powershell_exe,
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(service_script),
        "-Action",
        "Install",
        "-Broker",
        "mt5",
        "-Port",
        "8055",
        "-DryRun",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    stdout = proc.stdout

    # Verify that install command targets CLI run entrypoint
    assert has_dry_run_nssm_call(stdout, "install XauBot")
    assert "-m xau_bot.cli run --broker mt5 --port 8055 --log-json" in stdout
    assert has_dry_run_nssm_call(stdout, "set XauBot AppDirectory")
    assert has_dry_run_nssm_call(stdout, "set XauBot AppStdout")
    assert has_dry_run_nssm_call(stdout, "set XauBot AppStderr")
    assert has_dry_run_nssm_call(stdout, "set XauBot AppRotateFiles 1")
    assert has_dry_run_nssm_call(stdout, "set XauBot AppRestartDelay 5000")


def test_service_nssm_cli_arguments_parser_compatibility(
    scripts_dir: Path, powershell_exe: str
) -> None:
    """H-01 regression: service parameters must parse correctly in CLI main."""
    service_script = scripts_dir / "service-nssm.ps1"
    cmd = [
        powershell_exe,
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(service_script),
        "-Action",
        "Install",
        "-Broker",
        "mt5",
        "-Port",
        "8077",
        "-DryRun",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0

    # Extract command arguments passed to python
    match = re.search(r"nssm\.exe install XauBot \S+ -m xau_bot\.cli (.+)", proc.stdout)
    assert match is not None, "Install command line must match expected format"

    args_str = match.group(1).strip()
    raw_args = args_str.split()

    parser = build_parser()
    parsed = parser.parse_args(raw_args)

    assert parsed.command == "run"
    assert parsed.broker == "mt5"
    assert parsed.port == 8077
    assert parsed.log_json is True


def test_service_nssm_dry_run_actions(scripts_dir: Path, powershell_exe: str) -> None:
    service_script = scripts_dir / "service-nssm.ps1"
    for action in ["Start", "Stop", "Restart", "Remove"]:
        cmd = [
            powershell_exe,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(service_script),
            "-Action",
            action,
            "-ServiceName",
            "TestBot",
            "-DryRun",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert proc.returncode == 0
        assert has_dry_run_nssm_call(proc.stdout, f"{action.lower()} TestBot")


def test_service_nssm_status_with_invalid_nssm_path(scripts_dir: Path, powershell_exe: str) -> None:
    """N-04: Status action does not require NSSM binary and ignores invalid NssmPath."""
    service_script = scripts_dir / "service-nssm.ps1"
    cmd = [
        powershell_exe,
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(service_script),
        "-Action",
        "Status",
        "-ServiceName",
        "NonExistentServiceForTest",
        "-NssmPath",
        r"C:\non_existent_folder_xyz\nssm.exe",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    assert "Querying status for NonExistentServiceForTest" in proc.stdout


def test_service_nssm_error_propagation_on_failure(
    scripts_dir: Path, powershell_exe: str, tmp_path: Path
) -> None:
    """L-02: service-nssm.ps1 must propagate non-zero exit code when nssm fails."""
    service_script = scripts_dir / "service-nssm.ps1"
    shim_file = tmp_path / "nssm.cmd"
    shim_file.write_text("@echo off\r\nexit /b 2\r\n", encoding="utf-8")

    cmd = [
        powershell_exe,
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(service_script),
        "-Action",
        "Start",
        "-ServiceName",
        "TestFailingService",
        "-NssmPath",
        str(shim_file),
        "-SkipAdminCheck",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 2, (
        f"Script should exit with code 2, got {proc.returncode}. Output: {proc.stderr}"
    )


def test_service_nssm_warns_on_system_python_fallback(
    scripts_dir: Path, powershell_exe: str, tmp_path: Path
) -> None:
    """N-02: Install action warns when .venv is not found."""
    fake_scripts_dir = tmp_path / "scripts"
    fake_scripts_dir.mkdir(parents=True, exist_ok=True)
    target_script = fake_scripts_dir / "service-nssm.ps1"
    shutil.copy2(scripts_dir / "service-nssm.ps1", target_script)

    cmd = [
        powershell_exe,
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(target_script),
        "-Action",
        "Install",
        "-DryRun",
        "-SkipAdminCheck",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0
    combined = proc.stdout + proc.stderr
    assert "Local virtualenv (.venv) not found" in combined


def test_health_routes_integrity() -> None:
    """L-01 / AC-2005: Verify health check server routes integrity."""
    from xau_bot.server import HealthStatusState

    state = HealthStatusState()
    code_health, payload_health = state.get_health()
    assert code_health == 200
    assert "status" in payload_health
    assert "broker_connected" in payload_health
    assert "trading_halted" in payload_health

    code_status, payload_status = state.get_status()
    assert code_status == 200
    assert "positions" in payload_status
    assert "daily_pnl" in payload_status
    assert "kill_switches_active" in payload_status

    assert hasattr(HealthCheckHandler, "do_GET")
    assert hasattr(HealthCheckHandler, "_send_json")


def test_health_routes_real_http_requests() -> None:
    """L-03: Real HTTP requests against create_server on ephemeral port."""
    import threading
    import urllib.error
    import urllib.request

    from xau_bot.server import HealthStatusState, create_server

    state = HealthStatusState(broker_connected=True, trading_halted=False)

    server = create_server(host="127.0.0.1", port=0, state=state)
    port = server.server_address[1]

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        # 1. Test /health -> 200 OK
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "ok"
            assert data["broker_connected"] is True
            assert data["trading_halted"] is False

        # 2. Test /status -> 200 OK
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/status") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "positions" in data
            assert "daily_pnl" in data

        # 3. Test non-existent route -> 404
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/unknown_route")
        assert exc_info.value.code == 404
        err_body = json.loads(exc_info.value.read().decode("utf-8"))
        assert err_body == {"error": "not found"}
    finally:
        server.shutdown()
        server.server_close()


def test_runbook_contains_all_core_sections(docs_dir: Path) -> None:
    runbook_path = docs_dir / "RUNBOOK.md"
    assert runbook_path.exists(), "RUNBOOK.md must exist in docs/"

    content = runbook_path.read_text(encoding="utf-8")

    expected_sections = [
        "## 1. Visão Geral da Arquitetura em Produção",
        "## 2. Requisitos e Dimensionamento de VPS",
        "## 3. Provisionamento do Sistema Operacional",
        "## 4. Sincronização de Relógio de Alta Precisão (NTP)",
        "## 5. Prevenção de Reinicializações do Windows Update",
        "## 6. Instalação e Configuração do MetaTrader 5",
        "## 7. Instalação e Deploy do Código `xau-bot`",
        "## 8. Configuração de Variáveis de Ambiente e Segredos",
        "## 9. Gerenciamento do Serviço Windows (NSSM)",
        "## 10. Observabilidade, Health Check e Alertas",
        "## 11. Checklist Operacional Diário",
        "## 12. Plano de Resposta a Desastres (DRP)",
    ]

    for section in expected_sections:
        assert section in content, f"Missing section in RUNBOOK.md: {section}"

    assert "XauGuardian.mq5" in content
    assert "Common\\Files" in content
    assert "w32time" in content
    assert "SpecialPollInterval" in content
    assert "Autologon" in content
    assert "NoAutoRebootWithLoggedOnUsers" in content
    assert "service-nssm.ps1" in content
    assert "verify-vps.ps1" in content
    assert "/health" in content


def test_runbook_documents_core_settings_env_vars(docs_dir: Path) -> None:
    runbook_path = docs_dir / "RUNBOOK.md"
    content = runbook_path.read_text(encoding="utf-8")

    core_keys = [
        "mt5_server",
        "mt5_login",
        "mt5_password",
        "mt5_timeout_ms",
        "broker_server_tz",
        "guardian_common_dir",
        "guardian_calendar_max_age_h",
        "guardian_magic",
        "telegram_token",
        "telegram_chat_id",
        "server_host",
        "server_port",
        "max_daily_loss_pct",
        "max_drawdown_pct",
        "max_consecutive_losses",
        "max_daily_trades",
        "risk_per_trade_pct",
    ]

    for key in core_keys:
        assert key in Settings.model_fields
        env_var = f"XAU_{key.upper()}"
        assert env_var in content, (
            f"Environment variable {env_var} must be documented in RUNBOOK.md"
        )
