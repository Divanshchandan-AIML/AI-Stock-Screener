from pathlib import Path
import json
import csv
import shutil
from datetime import datetime, timezone


# ============================================================
# STAGE 26 - FINAL RELEASE AUDIT & PACKAGING
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "stage26"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REQUIRED ARTIFACTS FROM STAGES 21-25
# ============================================================

REQUIRED_ARTIFACTS = {
    "stage21": [
        "stage21_health_report.csv",
        "stage21_metrics.json",
        "stage21_monitoring_snapshot.csv",
        "stage21_paper_trades.csv",
        "stage21_release_manifest.json",
    ],

    "stage22": [
        "stage22_evaluation_report.csv",
        "stage22_metrics.json",
        "stage22_release_manifest.json",
    ],

    "stage23": [
        "stage23_health_report.csv",
        "stage23_metrics.json",
        "stage23_release_manifest.json",
    ],

    "stage24": [
        "stage24_health_report.csv",
        "stage24_metrics.json",
        "stage24_risk_controlled_trades.csv",
        "stage24_release_manifest.json",
    ],

    "stage25": [
        "stage25_health_report.csv",
        "stage25_metrics.json",
        "stage25_release_manifest.json",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_check(
    checks,
    category,
    name,
    passed,
    value,
    threshold,
    message
):
    checks.append({
        "timestamp": now_iso(),
        "category": category,
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "value": value,
        "threshold": threshold,
        "message": message,
    })


def check_csv_readable(path):
    try:
        with open(
            path,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as f:

            reader = csv.reader(f)

            rows = list(reader)

        return len(rows) > 0

    except Exception:
        return False


def check_json_readable(path):
    try:
        data = load_json(path)
        return isinstance(data, dict)

    except Exception:
        return False


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("STAGE 26 - FINAL RELEASE AUDIT & PACKAGING")
print("=" * 80)

print()
print("Project:")
print(PROJECT_ROOT)

print()
print("Audit output:")
print(OUTPUT_DIR)


# ============================================================
# CHECK ALL REQUIRED ARTIFACTS
# ============================================================

checks = []

existing_artifacts = []

missing_artifacts = []


for stage, filenames in REQUIRED_ARTIFACTS.items():

    stage_dir = DATA_DIR / stage

    for filename in filenames:

        path = stage_dir / filename

        exists = path.exists()

        if exists:
            existing_artifacts.append(path)
        else:
            missing_artifacts.append(path)

        add_check(
            checks,
            "ARTIFACT",
            f"{stage}_{filename}_exists",
            exists,
            str(path),
            "EXISTS",
            "Required release artifact must exist."
        )


# ============================================================
# CHECK FILE READABILITY
# ============================================================

for path in existing_artifacts:

    suffix = path.suffix.lower()

    if suffix == ".json":

        readable = check_json_readable(path)

        add_check(
            checks,
            "READABILITY",
            f"{path.name}_readable",
            readable,
            readable,
            True,
            "JSON artifact must be readable."
        )

    elif suffix == ".csv":

        readable = check_csv_readable(path)

        add_check(
            checks,
            "READABILITY",
            f"{path.name}_readable",
            readable,
            readable,
            True,
            "CSV artifact must be readable."
        )


# ============================================================
# LOAD STAGE 25 METRICS
# ============================================================

stage25_metrics_path = (
    DATA_DIR
    / "stage25"
    / "stage25_metrics.json"
)


stage25_metrics = None


if stage25_metrics_path.exists():

    try:

        stage25_metrics = load_json(
            stage25_metrics_path
        )

        stage25_loaded = True

    except Exception:

        stage25_loaded = False

else:

    stage25_loaded = False


add_check(
    checks,
    "FINAL_GATE",
    "stage25_metrics_loaded",
    stage25_loaded,
    stage25_loaded,
    True,
    "Stage 25 metrics must be readable."
)


# ============================================================
# VERIFY STAGE 25 PROMOTION DECISION
# ============================================================

if stage25_loaded:

    final_decision = stage25_metrics.get(
        "final_decision",
        ""
    )

    promotion_ready = (
        final_decision == "PROMOTION_READY"
    )

else:

    final_decision = ""
    promotion_ready = False


add_check(
    checks,
    "FINAL_GATE",
    "stage25_promotion_ready",
    promotion_ready,
    final_decision,
    "PROMOTION_READY",
    "Stage 25 must report PROMOTION_READY."
)


# ============================================================
# VERIFY PAPER-TRADING-ONLY
# ============================================================

if stage25_loaded:

    paper_only = (
        stage25_metrics.get(
            "paper_trading_only",
            False
        )
        is True
    )

else:

    paper_only = False


add_check(
    checks,
    "SAFETY",
    "paper_trading_only",
    paper_only,
    paper_only,
    True,
    "Paper trading must remain enabled."
)


# ============================================================
# VERIFY STAGE 25 CHECKS
# ============================================================

if stage25_loaded:

    stage25_checks = stage25_metrics.get(
        "checks",
        {}
    )

    stage25_total = int(
        stage25_checks.get("total", 0)
    )

    stage25_passed = int(
        stage25_checks.get("passed", 0)
    )

    stage25_failed = int(
        stage25_checks.get("failed", 0)
    )

else:

    stage25_total = 0
    stage25_passed = 0
    stage25_failed = 1


add_check(
    checks,
    "FINAL_GATE",
    "stage25_no_failed_checks",
    stage25_failed == 0,
    stage25_failed,
    0,
    "Stage 25 must have zero failed checks."
)


# ============================================================
# VERIFY RISK METRICS
# ============================================================

if stage25_loaded:

    metrics = stage25_metrics.get(
        "risk_controlled_metrics",
        {}
    )

    win_rate = float(
        metrics.get("win_rate", 0)
    )

    profit_factor = float(
        metrics.get("profit_factor", 0)
    )

    sharpe_ratio = float(
        metrics.get("sharpe_ratio", 0)
    )

    maximum_drawdown = float(
        metrics.get("maximum_drawdown", 0)
    )

else:

    win_rate = 0
    profit_factor = 0
    sharpe_ratio = 0
    maximum_drawdown = 0


add_check(
    checks,
    "RISK",
    "controlled_drawdown",
    maximum_drawdown >= -0.30,
    maximum_drawdown,
    -0.30,
    "Controlled maximum drawdown must be at least -30%."
)


add_check(
    checks,
    "PERFORMANCE",
    "controlled_win_rate",
    win_rate >= 0.50,
    win_rate,
    0.50,
    "Controlled win rate must be at least 50%."
)


add_check(
    checks,
    "PERFORMANCE",
    "controlled_profit_factor",
    profit_factor >= 1.00,
    profit_factor,
    1.00,
    "Controlled profit factor must be at least 1.00."
)


add_check(
    checks,
    "PERFORMANCE",
    "controlled_sharpe",
    sharpe_ratio >= 0.00,
    sharpe_ratio,
    0.00,
    "Controlled Sharpe ratio must be non-negative."
)


# ============================================================
# FINAL AUDIT RESULT
# ============================================================

total_checks = len(checks)

passed_checks = sum(
    1
    for check in checks
    if check["status"] == "PASS"
)

failed_checks = sum(
    1
    for check in checks
    if check["status"] == "FAIL"
)

pass_rate = (
    passed_checks / total_checks
    if total_checks
    else 0
)


if failed_checks == 0:

    audit_decision = "RELEASE_READY"

else:

    audit_decision = "REVIEW"


# ============================================================
# SAVE AUDIT REPORT
# ============================================================

health_report_path = (
    OUTPUT_DIR
    / "stage26_health_report.csv"
)


with open(
    health_report_path,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "timestamp",
            "category",
            "check",
            "status",
            "value",
            "threshold",
            "message",
        ]
    )

    writer.writeheader()
    writer.writerows(checks)


# ============================================================
# BUILD FINAL RELEASE SUMMARY
# ============================================================

release_summary = {

    "stage": "26",

    "stage_name":
        "Final Release Audit & Packaging",

    "generated_at":
        now_iso(),

    "audit_decision":
        audit_decision,

    "promotion_decision":
        final_decision,

    "paper_trading_only":
        paper_only,

    "stage25": {

        "checks_total":
            stage25_total,

        "checks_passed":
            stage25_passed,

        "checks_failed":
            stage25_failed,

        "win_rate":
            win_rate,

        "profit_factor":
            profit_factor,

        "sharpe_ratio":
            sharpe_ratio,

        "maximum_drawdown":
            maximum_drawdown,

    },

    "stage26_checks": {

        "total":
            total_checks,

        "passed":
            passed_checks,

        "failed":
            failed_checks,

        "pass_rate":
            pass_rate,

    },

    "required_artifacts":
        len(existing_artifacts)
        + len(missing_artifacts),

    "artifacts_found":
        len(existing_artifacts),

    "artifacts_missing":
        len(missing_artifacts),

    "missing_files": [
        str(path)
        for path in missing_artifacts
    ],
}


# ============================================================
# SAVE RELEASE SUMMARY
# ============================================================

summary_path = (
    OUTPUT_DIR
    / "stage26_release_summary.json"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        release_summary,
        f,
        indent=2
    )


# ============================================================
# CREATE RELEASE MANIFEST
# ============================================================

manifest = {

    "release_name":
        "AI_Stock_Screener_Final",

    "release_stage":
        "26",

    "generated_at":
        now_iso(),

    "audit_decision":
        audit_decision,

    "promotion_decision":
        final_decision,

    "paper_trading_only":
        True,

    "live_trading":
        "DISABLED",

    "stages_validated": [
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
    ],

    "artifacts": {
        "health_report":
            str(health_report_path),

        "release_summary":
            str(summary_path),
    },

    "audit": {
        "checks_total":
            total_checks,

        "checks_passed":
            passed_checks,

        "checks_failed":
            failed_checks,

        "pass_rate":
            pass_rate,
    },
}


manifest_path = (
    OUTPUT_DIR
    / "stage26_release_manifest.json"
)


with open(
    manifest_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


# ============================================================
# CREATE RELEASE DIRECTORY
# ============================================================

release_dir = (
    OUTPUT_DIR
    / "release_package"
)


release_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# COPY IMPORTANT FINAL ARTIFACTS
# ============================================================

files_to_package = [

    DATA_DIR
    / "stage21"
    / "stage21_paper_trades.csv",

    DATA_DIR
    / "stage21"
    / "stage21_metrics.json",

    DATA_DIR
    / "stage22"
    / "stage22_evaluation_report.csv",

    DATA_DIR
    / "stage22"
    / "stage22_metrics.json",

    DATA_DIR
    / "stage23"
    / "stage23_health_report.csv",

    DATA_DIR
    / "stage23"
    / "stage23_metrics.json",

    DATA_DIR
    / "stage24"
    / "stage24_risk_controlled_trades.csv",

    DATA_DIR
    / "stage24"
    / "stage24_metrics.json",

    DATA_DIR
    / "stage25"
    / "stage25_health_report.csv",

    DATA_DIR
    / "stage25"
    / "stage25_metrics.json",

    health_report_path,

    summary_path,

    manifest_path,
]


packaged_files = []


for source in files_to_package:

    if source.exists():

        destination = (
            release_dir
            / source.name
        )

        shutil.copy2(
            source,
            destination
        )

        packaged_files.append(
            destination.name
        )


# ============================================================
# PACKAGE MANIFEST
# ============================================================

package_manifest = {

    "release":
        "AI_Stock_Screener_Final",

    "created_at":
        now_iso(),

    "audit_decision":
        audit_decision,

    "promotion_decision":
        final_decision,

    "paper_trading_only":
        True,

    "live_trading":
        "DISABLED",

    "files":
        packaged_files,

}


package_manifest_path = (
    release_dir
    / "PACKAGE_MANIFEST.json"
)


with open(
    package_manifest_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        package_manifest,
        f,
        indent=2
    )


# ============================================================
# FINAL TERMINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("STAGE 26 FINAL RELEASE AUDIT")
print("=" * 80)

print()
print(
    f"Artifacts found       : "
    f"{len(existing_artifacts)}"
)

print(
    f"Artifacts missing     : "
    f"{len(missing_artifacts)}"
)

print()
print(
    f"Stage 25 checks       : "
    f"{stage25_passed}/{stage25_total}"
)

print(
    f"Stage 25 failed       : "
    f"{stage25_failed}"
)

print()
print(
    f"Win rate              : "
    f"{win_rate:.2%}"
)

print(
    f"Profit factor         : "
    f"{profit_factor:.4f}"
)

print(
    f"Sharpe ratio          : "
    f"{sharpe_ratio:.4f}"
)

print(
    f"Maximum drawdown      : "
    f"{maximum_drawdown:.2%}"
)

print()
print(
    f"Stage 26 checks       : "
    f"{passed_checks}/{total_checks}"
)

print(
    f"Stage 26 pass rate    : "
    f"{pass_rate:.2%}"
)

print()
print(
    f"PAPER TRADING ONLY    : "
    f"{'YES' if paper_only else 'NO'}"
)

print(
    "LIVE TRADING          : DISABLED"
)

print()
print(
    f"PROMOTION DECISION    : "
    f"{final_decision}"
)

print(
    f"AUDIT DECISION        : "
    f"{audit_decision}"
)

print()
print("Health report:")
print(health_report_path)

print()
print("Release summary:")
print(summary_path)

print()
print("Release manifest:")
print(manifest_path)

print()
print("Release package:")
print(release_dir)

print()
print("=" * 80)
print("STAGE 26 COMPLETE")
print("=" * 80)