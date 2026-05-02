# 🚀 소켓 기반 분산 학습 파라미터 서버

Python TCP 소켓을 이용해 **분산 학습의 핵심 아키텍처인 Parameter Server 를 밑바닥부터 구현**한 프로젝트입니다. PyTorch 의 `DistributedDataParallel` 같은 고급 라이브러리에 의존하지 않고, 네트워크 통신과 동기화 로직을 직접 구현하여 MLsys 의 본질을 이해하는 데 목적이 있습니다.

## 📋 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **아키텍처** | Parameter Server (Centralized All-Reduce) |
| **통신 프로토콜** | TCP/IP Socket (Python `socket` 모듈) |
| **직렬화** | `pickle` + 4 바이트 길이 헤더 |
| **동기화** | Barrier Synchronization |
| **최대 Worker 수** | 테스트 결과 **8 개 이상 가능** 확인 |

## 🏗️ 시스템 아키텍처

```
                    ┌─────────────────┐
                    │   Master Node   │
                    │ (Parameter Server)│
                    │  - Global Weights│
                    │  - Gradient Avg  │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
    ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
    │  Worker 0   │   │  Worker 1   │   │  Worker N   │
    │  - Local Data│   │  - Local Data│   │  - Local Data│
    │  - Compute Grad│  │  - Compute Grad│  │  - Compute Grad│
    └─────────────┘   └─────────────┘   └─────────────┘
```

### 동작 파이프라인

1. **Forward & Backward**: 각 Worker 가 로컬 데이터로 기울기 계산
2. **Serialize & Send**: `pickle` 직렬화 → TCP 소켓으로 Master 에 전송
3. **Barrier & Aggregate**: Master 가 모든 Worker 의 기울기 수집 후 평균 (All-Reduce)
4. **Update & Broadcast**: Master 가 전역 가중치 업데이트 → Worker 에 브로드캐스트
5. **Sync**: Worker 가 새 가중치로 모델 동기화

## 🛠️ 설치 및 실행

### 필요 패키지
```bash
pip install numpy
```

### 자동 실행 (권장)
```bash
# 기본: 3 개 Worker, 5 에폭
./run_distributed.sh

# 커스텀: 5 개 Worker, 10 에폭
./run_distributed.sh 5 10
```

### 수동 실행
```bash
# 터미널 1: Master 서버 시작 (예: 3 개 Worker 예상)
python master.py 3

# 터미널 2~4: Worker 시작 (각각 다른 ID)
python worker.py 0
python worker.py 1
python worker.py 2
```

### 환경 변수
| 변수 | 설명 | 기본값 |
|------|------|--------|
| `DL_EPOCHS` | 학습 에폭 수 | 5 |
| `MODEL_SIZE` | 모델 가중치 차원 | 1000 |
| `LEARNING_RATE` | 학습률 | 0.01 |
| `BATCH_SIZE` | 미니 배치 크기 | 32 |

## 📊 성능 벤치마크

**테스트 환경**: 모델 1000 차원, 5 에폭 기준

| Worker 수 | 총 시간 (초) | 에폭당 시간 (초) |
|-----------|--------------|------------------|
| 2         | 2.2          | 0.44             |
| 3         | 2.6          | 0.52             |
| 5         | 3.5          | 0.70             |
| 8         | 4.9          | 0.98             |

> 💡 **인사이트**: Worker 수가 증가할수록 Barrier 대기 시간이 늘어나지만, 더 많은 데이터를 병렬 처리할 수 있는 트레이드오프가 있습니다.

## 🎯 이 프로젝트로 어필할 수 있는 역량

### 역량기술서 작성 예시

> **[경험] Python Socket 기반 분산 학습 파라미터 서버 구현**
> 
> 대규모 분산 학습의 네트워크 병목을 구조적으로 이해하기 위해, PyTorch 라이브러리에 의존하지 않고 밑바닥부터 분산 환경을 시뮬레이션했습니다. Python TCP 소켓 통신을 이용해 다중 Worker 노드가 미니 배치 학습 후 기울기 (Gradient) 텐서를 직렬화하여 Master 노드로 전송하고, Master 가 이를 평균 내어 동기화하는 파라미터 서버 아키텍처를 직접 구현했습니다.
> 
> **주요 성과:**
> - Barrier 동기화를 통한 All-Reduce 연산 구현으로 분산 학습의 핵심 메커니즘 이해
> - 8 개 이상의 Worker 를 안정적으로 동기화하며 확장성 검증
> - 모델 크기 대비 통신 오버헤드의 영향을 체감적으로 분석
> - 향후 K8s 환경에서 vLLM 추론 스케줄링 및 병목 최적화 실험을 수행할 수 있는 시스템 아키텍처 지반 확보

### 면접 대비 포인트

1. **왜 Parameter Server 아키텍처를 선택했나요?**
   - 분산 학습의 가장 직관적인 구조로, 통신과 계산을 명확히 분리할 수 있어 학습용으로 적합합니다. 실제 프로덕션에서는 Ring All-Reduce 등을 고려할 수 있습니다.

2. **동기식 vs 비동기식 중 어떤 방식을 썼나요?**
   - 동기식 (Barrier) 을 사용했습니다. 모든 Worker 의 기울기가 모일 때까지 대기하므로 안정적인 수렴을 보장하지만, 느린 Worker(Straggler) 문제가 발생할 수 있습니다.

3. **네트워크 병목은 어떻게 해결했나요?**
   - 4 바이트 길이 헤더 + pickle 직렬화로 효율적인 프로토콜을 설계했습니다. 실제 프로덕션에서는 gRPC, NCCL, 또는 Tensor 압축 (Quantization) 등을 고려할 수 있습니다.

## 🔮 확장 아이디어

- [ ] **Ring All-Reduce**: Parameter Server 없는 P2P 통신 구현
- [ ] **비동기 업데이트**: Barrier 제거로 Straggler 문제 해결
- [ ] **Gradient Compression**: 16-bit 양자화로 통신량 50% 감소
- [ ] **Docker 컨테이너화**: `docker-compose` 로 멀티 노드 클러스터 시뮬레이션
- [ ] **PyTorch 통합**: 실제 신경망 모델로 확장

## 📁 파일 구조

```
/workspace
├── master.py              # 파라미터 서버 (Master)
├── worker.py              # 학습 Worker (클라이언트)
├── run_distributed.sh     # 자동 실행 스크립트
├── test_benchmark.py      # 성능 벤치마크 스크립트
└── README.md              # 이 문서
```

---

**작성자**: [본인 이름]  
**작성일**: 2025 년  
**연락처**: [이메일/ GitHub]
