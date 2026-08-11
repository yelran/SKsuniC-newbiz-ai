"""
경로 정의 · 파트 모듈 로더

■ 이 파일의 역할
  app/ 안의 모든 경로를 여기서만 정의한다. main.py나 각 파트 코드가
  Path(__file__).parent / ".." 같은 걸 직접 만들면, 파일이 폴더를 옮길 때마다
  전부 깨진다. 경로가 필요하면 여기서 import 한다.

■ 왜 importlib 로더가 여기 있나
  파트 파일 이름에 하이픈이 있다(F2-5.py, F3-4.py). 파이썬 식별자로 쓸 수 없어서
  `import F2-5`가 안 된다. 파일 경로로 직접 로드해야 하고, 그건 경로 문제라
  경로를 아는 이 파일이 맡는다.

■ 사용법
    from core.paths import DATA_DIR, DB_PATH, load_module, try_load_module

    f2_5 = load_module("f2_5", "F2-5.py")          # 없으면 FileNotFoundError
    f5   = try_load_module("f5", "F5.py")          # 없으면 None ('연결 대기')
"""

import importlib.util
import sys
from pathlib import Path

# core/paths.py → core/ → app/
CORE = Path(__file__).resolve().parent
BASE = CORE.parent

# ── 파트 폴더 ────────────────────────────────────────────────────────
UPLOAD_DIR = CORE / "upload"        # F1        데이터 업로드
PROFILE_DIR = CORE / "profile"      # F2        조직 역량 프로필
RECOMMEND_DIR = CORE / "recommend"  # F2-2·F3-1·F3-3·F4-3  역량 기반 추천형
IDEA_FIT_DIR = CORE / "idea_fit"    # F4-1·F4-2 아이디어 적합도 판단형
RESULT_DIR = CORE / "result"        # F5        결과 대시보드
SCORING_DIR = CORE / "scoring"      # F2-4·F2-5·F3-2·F3-4  공용 채점

# ── 데이터 · 자산 ────────────────────────────────────────────────────
DATA_DIR = BASE / "data"            # 리포에 포함되는 데이터
ASSETS_DIR = BASE / "assets"
SAMPLES_DIR = BASE / "samples"      # 🔒 개발용 조직 데이터 (없어도 실행됨)
#
# 산출물 폴더(outputs/)는 두지 않는다.
#   업로드한 조직 데이터로 만든 프로필을 고정 경로에 저장하면, Streamlit Cloud는
#   컨테이너 하나를 모든 사용자가 공유하므로 사용자 A의 인력정보·특허목록이
#   사용자 B에게 그대로 보인다. 프로필은 st.session_state에만 두고
#   창을 닫으면 사라지게 한다. → main.py load_org_profile() 참고

DB_PATH = DATA_DIR / "신사업_DB.xlsx"
CAPABILITY_DEF_PATH = DATA_DIR / "표준역량_정의.csv"
CAPABILITY_MAP_PATH = DATA_DIR / "역량어휘_매핑.csv"
LOGO_PATH = ASSETS_DIR / "logo.png"

# 모듈을 찾을 때 뒤지는 순서. 파트 폴더를 옮기거나 팀원이 엉뚱한 폴더에
# 올려도 앱이 죽지 않게 전체를 훑는다.
MODULE_DIRS = [UPLOAD_DIR, PROFILE_DIR, RECOMMEND_DIR,
               IDEA_FIT_DIR, RESULT_DIR, SCORING_DIR]

# 파트 폴더를 전부 sys.path에 올린다.
#   F4-2(idea_fit)와 F4-3(recommend)이 `from f4_common import ...`처럼
#   하이픈 없는 공용 파일을 평범하게 import한다. importlib로 파일 경로를
#   찍어 로드하면 그 폴더가 sys.path에 안 들어가서 ModuleNotFoundError가 난다.
for _d in MODULE_DIRS:
    _s = str(_d)
    if _d.is_dir() and _s not in sys.path:
        sys.path.append(_s)


def module_path(filename: str) -> Path | None:
    """파트 파일의 실제 위치. 없으면 None."""
    for d in MODULE_DIRS:
        p = d / filename
        if p.exists():
            return p
    return None


def load_module(name: str, filename: str):
    """하이픈 있는 파트 파일을 이름으로 로드한다.

    같은 name이 이미 로드돼 있으면 재사용한다. F3-4가 내부에서 F2-5·F3-2를
    또 로드하기 때문에, 캐시가 없으면 같은 파일이 여러 번 실행되면서
    배점 상수가 중복 생성된다.
    """
    if name in sys.modules:
        return sys.modules[name]

    path = module_path(filename)
    if path is None:
        raise FileNotFoundError(
            f"{filename}을 찾을 수 없다. 찾아본 곳: "
            + ", ".join(str(d) for d in MODULE_DIRS))

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # 순환 참조 대비, exec 전에 등록
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)  # 실패한 모듈을 캐시에 남기지 않는다
        raise
    return mod


def try_load_module(name: str, filename: str):
    """없거나 로드에 실패하면 None. 아직 안 올라온 파트를 '연결 대기'로 두기 위한 것."""
    try:
        return load_module(name, filename)
    except Exception:
        return None
