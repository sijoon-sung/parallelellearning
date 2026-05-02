"""
분산 학습 Worker Node
- 로컬 데이터로 기울기 계산 후 Master 에 전송
- Master 로부터 업데이트된 가중치를 받아 모델 동기화
"""
import socket
import pickle
import struct
import numpy as np
import os
import sys
import time

# 설정
HOST = 'localhost'
PORT = 5000
MODEL_SIZE = int(os.environ.get("MODEL_SIZE", 1000))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))

class Worker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.weights = None
        
        # 각 Worker 마다 다른 데이터 분포 (비독립동분포 시뮬레이션)
        np.random.seed(worker_id + 42)
        self.local_data_x = np.random.randn(BATCH_SIZE, MODEL_SIZE).astype(np.float32)
        # 단순 선형 회귀 문제: y = sum(x) + noise
        self.local_data_y = np.sum(self.local_data_x, axis=1).astype(np.float32)
        
    def compute_gradient(self, weights):
        """로컬 데이터로 기울기 계산 (MSE 손실함수 기준)"""
        predictions = np.dot(self.local_data_x, weights)
        errors = predictions - self.local_data_y
        gradient = np.dot(self.local_data_x.T, errors) / len(errors)
        return gradient.astype(np.float32)

    def _recv_all(self, conn, length):
        """지정된 길이만큼 데이터를 모두 수신"""
        data = b''
        while len(data) < length:
            packet = conn.recv(length - len(data))
            if not packet:
                return None
            data += packet
        return data

    def run(self, epochs):
        print(f"[Worker-{self.worker_id}] 서버에 연결 시도...")
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((HOST, PORT))
            sock.settimeout(30)  # 30 초 타임아웃
            print(f"[Worker-{self.worker_id}] ✅ 서버와 연결됨!")
            
            for epoch in range(epochs):
                start_time = time.time()
                
                # 1. 현재 가중치 수신 (Master 가 먼저 보냄)
                raw_length = self._recv_all(sock, 4)
                if not raw_length:
                    break
                msg_len = struct.unpack('!I', raw_length)[0]
                weights_data = self._recv_all(sock, msg_len)
                self.weights = pickle.loads(weights_data)
                
                # 2. 로컬 데이터로 기울기 계산
                gradient = self.compute_gradient(self.weights)
                
                # 3. 기울기 전송
                data = pickle.dumps(gradient)
                length = len(data)
                sock.sendall(struct.pack('!I', length) + data)
                
                elapsed = time.time() - start_time
                print(f"[Worker-{self.worker_id}] 에폭 {epoch+1}/{epochs} 완료 ({elapsed:.3f}초)")
                
            print(f"[Worker-{self.worker_id}] ✅ 학습 종료")
            
        except socket.timeout:
            print(f"[Worker-{self.worker_id}] ⚠️ 연결 타임아웃")
        except Exception as e:
            print(f"[Worker-{self.worker_id}] ❌ 오류: {e}")
        finally:
            sock.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python worker.py [워커 ID]")
        sys.exit(1)
        
    worker_id = int(sys.argv[1])
    epochs = int(os.environ.get("DL_EPOCHS", 5))
    
    print(f"[Worker-{worker_id}] 설정: Epochs={epochs}, ModelSize={MODEL_SIZE}, BatchSize={BATCH_SIZE}")
    worker = Worker(worker_id)
    worker.run(epochs)
