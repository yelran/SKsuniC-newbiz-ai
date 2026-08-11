def match_capabilities(org_caps: list, required_caps: list) -> dict:

    org_set = set(org_caps)
    required_set = set(required_caps)
    matched = list(org_set & required_set)
    missing = list(required_set - org_set)
    return {"matched": matched, "missing": missing}


if __name__ == "__main__":
    print("=" * 60)
    print("단독 실행 데모")
    print("=" * 60)

    cases = {
        "완전일치": (["영상진단", "정량계측", "클라우드"], ["영상진단", "정량계측", "클라우드"]),
        "완전불일치": (["영상진단"], ["음성인식", "위치기반", "블록체인"]),
        "빈 리스트": ([], ["영상진단"]),
    }

    for name, (org, req) in cases.items():
        result = match_capabilities(org, req)
        print(f"\n[{name}]")
        print("  org_caps      :", org)
        print("  required_caps :", req)
        print("  matched       :", result["matched"])
        print("  missing       :", result["missing"])
        
