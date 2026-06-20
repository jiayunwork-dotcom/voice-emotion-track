import requests
import json
import time
import os

API_BASE = "http://localhost:8000"


def test_health():
    print("=" * 50)
    print("测试1: 健康检查")
    resp = requests.get(f"{API_BASE}/api/health", timeout=5)
    print(f"  状态码: {resp.status_code}")
    assert resp.status_code == 200
    print("  ✓ 通过")


def test_batch_submit_and_process():
    print("\n" + "=" * 50)
    print("测试2: 批量提交和处理")
    
    batch_config = {
        "batch_name": "测试批次_自动化测试",
        "dimensions": ["emotion_distribution", "valence_trend", "arousal_pattern", "speaker_similarity"],
        "baseline_file": "test_1.wav"
    }
    
    files = []
    for i in range(1, 4):
        filepath = f"test_audio/test_{i}.wav"
        files.append(("files", (f"test_{i}.wav", open(filepath, "rb"))))
    
    params = {"batch_config": json.dumps(batch_config)}
    resp = requests.post(f"{API_BASE}/api/batch/submit", files=files, params=params, timeout=30)
    print(f"  提交状态码: {resp.status_code}")
    assert resp.status_code == 200
    
    result = resp.json()
    batch_id = result["batch_id"]
    print(f"  批次ID: {batch_id}")
    
    for f in files:
        f[1][1].close()
    
    print("\n测试3: 查询状态")
    completed = False
    for attempt in range(20):
        status_resp = requests.get(f"{API_BASE}/api/batch/{batch_id}/status", timeout=5)
        status = status_resp.json()
        pct = status["progress"]["percentage"]
        print(f"  进度: {pct:.1f}% ({status['progress']['completed']}/{status['progress']['total']})")
        if pct >= 100:
            completed = True
            break
        time.sleep(3)
    
    assert completed, "处理超时"
    print("  ✓ 通过")
    
    return batch_id


def test_report(batch_id):
    print("\n" + "=" * 50)
    print("测试4: 获取报告")
    
    resp = requests.get(f"{API_BASE}/api/batch/{batch_id}/report", timeout=10)
    print(f"  报告状态码: {resp.status_code}")
    assert resp.status_code == 200
    
    report = resp.json()
    meta = report["meta"]
    print(f"  批次名: {meta['batch_name']}")
    print(f"  总文件数: {meta['total_files']}")
    print(f"  成功数: {meta['success_count']}")
    print(f"  失败数: {meta['failed_count']}")
    
    comparison = report["comparison"]
    for dim in ["emotion_distribution", "valence_trend", "arousal_pattern", "speaker_similarity"]:
        assert dim in comparison, f"缺少维度: {dim}"
        print(f"  ✓ {dim} 维度已生成")
    
    if "emotion_distribution" in comparison:
        ed = comparison["emotion_distribution"]
        assert "distributions" in ed
        assert "js_divergence_matrix" in ed
        assert "most_divergent_pair" in ed
        if "baseline_deviations" in ed:
            print("  ✓ 基线偏差已计算")
    
    if "valence_trend" in comparison:
        vt = comparison["valence_trend"]
        assert "trends" in vt
        assert "anomalous_files" in vt
    
    print("  ✓ 报告结构完整")
    
    return report


def test_csv_export(batch_id):
    print("\n" + "=" * 50)
    print("测试5: CSV导出")
    
    resp = requests.get(f"{API_BASE}/api/batch/{batch_id}/report", params={"format": "csv"}, timeout=10)
    print(f"  CSV状态码: {resp.status_code}")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers.get("content-type", "")
    
    csv_content = resp.text
    lines = csv_content.split("\n")
    print(f"  CSV行数: {len(lines)}")
    assert len(lines) > 5
    print("  ✓ CSV导出成功")


def test_error_cases():
    print("\n" + "=" * 50)
    print("测试6: 错误处理")
    
    batch_config = {
        "batch_name": "测试",
        "dimensions": ["emotion_distribution"]
    }
    params = {"batch_config": json.dumps(batch_config)}
    
    # 测试: 测试文件数量不足
    print("  6.1 测试文件数量不足(1个文件)")
    f = open("test_audio/test_1.wav", "rb")
    files = [("files", ("test1.wav", f))]
    resp = requests.post(f"{API_BASE}/api/batch/submit", files=files, params=params, timeout=10)
    f.close()
    print(f"    状态码: {resp.status_code} (预期400)")
    assert resp.status_code == 400
    
    # 测试: 测试文件格式错误
    print("  6.2 测试不支持的文件格式")
    f1 = open("test_audio/test_1.wav", "rb")
    files = [
        ("files", ("test1.wav", f1)),
        ("files", ("test2.txt", b"not audio"))
    ]
    resp = requests.post(f"{API_BASE}/api/batch/submit", files=files, params=params, timeout=10)
    f1.close()
    print(f"    状态码: {resp.status_code} (预期400)")
    assert resp.status_code == 400
    
    # 测试: 测试不存在的批次
    print("  6.3 测试不存在的批次")
    resp = requests.get(f"{API_BASE}/api/batch/nonexistent/report", timeout=5)
    print(f"    状态码: {resp.status_code} (预期404)")
    assert resp.status_code == 404
    
    # 测试: 测试无效维度
    print("  6.4 测试无效维度")
    batch_config_bad = {
        "batch_name": "测试",
        "dimensions": ["invalid_dim"]
    }
    params_bad = {"batch_config": json.dumps(batch_config_bad)}
    f1 = open("test_audio/test_1.wav", "rb")
    f2 = open("test_audio/test_2.wav", "rb")
    files = [
        ("files", ("test1.wav", f1)),
        ("files", ("test2.wav", f2))
    ]
    resp = requests.post(f"{API_BASE}/api/batch/submit", files=files, params=params_bad, timeout=10)
    f1.close()
    f2.close()
    print(f"    状态码: {resp.status_code} (预期400)")
    assert resp.status_code == 400
    
    print("  ✓ 所有错误处理测试通过")


def main():
    try:
        test_health()
        batch_id = test_batch_submit_and_process()
        test_report(batch_id)
        test_csv_export(batch_id)
        test_error_cases()
        
        print("\n" + "=" * 50)
        print("🎉 所有测试通过!")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
