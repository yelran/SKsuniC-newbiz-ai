import importlib.util
import sys
from pathlib import Path

CORE = Path(__file__).resolve().parent
BASE = CORE.parent

# ── 폴더 ────────────────────────────────────────────────────────
UPLOAD_DIR = CORE / "upload"        # 데이터 업로드
PROFILE_DIR = CORE / "profile"      # 조직 역량 프로필
RECOMMEND_DIR = CORE / "recommend"  # 역량 기반 추천형
IDEA_FIT_DIR = CORE / "idea_fit"    # 아이디어 적합도 판단형
RESULT_DIR = CORE / "result"        # 결과 대시보드
SCORING_DIR = CORE / "scoring"      # 공용 채점

# ── 데이터 · 자산 ────────────────────────────────────────────────────
DATA_DIR = BASE / "data"            
ASSETS_DIR = BASE / "assets"
SAMPLES_DIR = BASE / "samples"      


DB_PATH = DATA_DIR / "신사업_DB.xlsx"
CAPABILITY_DEF_PATH = DATA_DIR / "표준역량_정의.csv"
CAPABILITY_MAP_PATH = DATA_DIR / "역량어휘_매핑.csv"
LOGO_PATH = ASSETS_DIR / "logo.png"


MODULE_DIRS = [UPLOAD_DIR, PROFILE_DIR, RECOMMEND_DIR,
               IDEA_FIT_DIR, RESULT_DIR, SCORING_DIR]


for _d in MODULE_DIRS:
    _s = str(_d)
    if _d.is_dir() and _s not in sys.path:
        sys.path.append(_s)


def module_path(filename: str) -> Path | None:
    for d in MODULE_DIRS:
        p = d / filename
        if p.exists():
            return p
    return None


def load_module(name: str, filename: str):
    if name in sys.modules:
        return sys.modules[name]

    path = module_path(filename)
    if path is None:
        raise FileNotFoundError(
            f"{filename}을 찾을 수 없다. 찾아본 곳: "
            + ", ".join(str(d) for d in MODULE_DIRS))

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)  
        raise
    return mod


def try_load_module(name: str, filename: str):
    try:
        return load_module(name, filename)
    except Exception:
        return None
