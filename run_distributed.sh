#!/bin/bash
# =============================================================================
# 분산 학습 파라미터 서버 자동 실행 스크립트
# 사용법: ./run_distributed.sh [워커 수] [에폭 수]
# 예시:  ./run_distributed.sh 5 10   (5 개 워커, 10 에폭)
# =============================================================================

NUM_WORKERS=${1:-3}  # 기본 3 개
EPOCHS=${2:-5}       # 기본 5 에폭

echo "============================================================"
echo "🚀 분산 학습 파라미터 서버 자동 시작"
echo "============================================================"
echo "  - 워커 수: $NUM_WORKERS 개"
echo "  - 에폭 수: $EPOCHS"
echo "  - 모델 크기: ${MODEL_SIZE:-1000} 차원"
echo "============================================================"
echo ""

# 환경 변수 설정
export DL_EPOCHS=$EPOCHS
export NUM_WORKERS=$NUM_WORKERS
export MODEL_SIZE=${MODEL_SIZE:-1000}
export LEARNING_RATE=${LEARNING_RATE:-0.01}
export BATCH_SIZE=${BATCH_SIZE:-32}

# 기존 프로세스 정리
pkill -f "python.*master.py" 2>/dev/null || true
pkill -f "python.*worker.py" 2>/dev/null || true
sleep 1

# Master 서버 백그라운드 실행
echo "[System] 📡 Master 서버 시작..."
python master.py $NUM_WORKERS > master.log 2>&1 &
MASTER_PID=$!
sleep 2

# Master 가 정상적으로 시작되었는지 확인
if ! kill -0 $MASTER_PID 2>/dev/null; then
    echo "❌ Master 서버가 시작되지 않았습니다. master.log 를 확인하세요."
    cat master.log
    exit 1
fi

# Worker 들 백그라운드 실행
echo "[System] 🔧 $NUM_WORKERS 개의 Worker 시작..."
WORKER_PIDS=()
for i in $(seq 0 $((NUM_WORKERS-1))); do
    python worker.py $i > worker_$i.log 2>&1 &
    WORKER_PIDS+=($!)
    echo "  - Worker-$i 시작 (PID: ${WORKER_PIDS[-1]})"
    sleep 0.1  # 동시 연결 폭주 방지
done

echo ""
echo "[System] ⏳ 학습 진행 중... ($EPOCHS 에폭)"
echo "------------------------------------------------------------"

# 모든 Worker 프로세스가 끝날 때까지 대기
FAILED=0
for pid in "${WORKER_PIDS[@]}"; do
    if ! wait $pid; then
        FAILED=1
    fi
done

echo "------------------------------------------------------------"

if [ $FAILED -eq 0 ]; then
    echo "[System] ✅ 모든 Worker 가 학습을 완료했습니다!"
else
    echo "[System] ⚠️ 일부 Worker 에서 오류가 발생했습니다."
fi

echo ""
echo "=== Master 로그 요약 ==="
grep -E "(Worker.*연결|✅|Round.*집계)" master.log | tail -15

echo ""
echo "=== Worker 별 소요 시간 ==="
for i in $(seq 0 $((NUM_WORKERS-1))); do
    if [ -f worker_$i.log ]; then
        LAST_LINE=$(tail -3 worker_$i.log | grep -E "에폭.*완료" | tail -1)
        if [ -n "$LAST_LINE" ]; then
            echo "  Worker-$i: $LAST_LINE"
        fi
    fi
done

# Master 프로세스 종료
kill $MASTER_PID 2>/dev/null || true
wait $MASTER_PID 2>/dev/null || true

echo ""
echo "============================================================"
echo "✅ 분산 학습 완료!"
echo "============================================================"
