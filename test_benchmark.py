"""
분산 학습 성능 벤치마크 스크립트
- 다양한 Worker 수에 따른 학습 시간 측정
"""
import subprocess
import time
import os

def run_benchmark(num_workers, epochs=5):
    print(f"\n{'='*60}")
    print(f"🔬 벤치마크: Worker {num_workers}개, Epochs {epochs}")
    print('='*60)
    
    # 환경 변수 설정
    env = os.environ.copy()
    env['DL_EPOCHS'] = str(epochs)
    env['NUM_WORKERS'] = str(num_workers)
    env['MODEL_SIZE'] = '1000'
    
    start_time = time.time()
    
    # Master 시작
    master_proc = subprocess.Popen(
        ['python', 'master.py', str(num_workers)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env
    )
    
    time.sleep(1.5)  # Master 가 준비될 때까지 대기
    
    # Worker 들 시작
    worker_procs = []
    for i in range(num_workers):
        proc = subprocess.Popen(
            ['python', 'worker.py', str(i)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        worker_procs.append(proc)
        time.sleep(0.05)
    
    # 모든 Worker 가 끝날 때까지 대기
    for proc in worker_procs:
        proc.wait()
    
    # Master 종료
    master_proc.terminate()
    master_proc.wait(timeout=2)
    
    elapsed = time.time() - start_time
    print(f"✅ 총 소요 시간: {elapsed:.2f}초")
    print(f"   Worker 당 평균 시간: {elapsed/epochs:.2f}초/에폭")
    
    return elapsed

if __name__ == "__main__":
    print("🚀 분산 학습 성능 벤치마크 시작")
    print("모델 크기: 1000 차원, 에폭: 5")
    
    results = []
    for n in [2, 3, 5, 8]:
        try:
            elapsed = run_benchmark(n, epochs=5)
            results.append((n, elapsed))
        except Exception as e:
            print(f"❌ Worker {n}개 테스트 실패: {e}")
    
    print("\n" + "="*60)
    print("📊 벤치마크 결과 요약")
    print("="*60)
    print(f"{'Worker 수':<12} | {'총 시간 (초)':<15} | {'에폭당 시간 (초)':<15}")
    print("-"*45)
    for n, t in results:
        print(f"{n:<12} | {t:<15.2f} | {t/5:<15.2f}")
    print("="*60)
