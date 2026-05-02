"""
분산 학습 파라미터 서버 (Master Node)
- 여러 Worker 로부터 기울기를 수집하여 평균낸 후 가중치 업데이트
- Barrier 동기화를 통해 모든 Worker 의 기울기가 모일 때까지 대기
"""
import socket
import pickle
import struct
import threading
import numpy as np
import os
import sys

# 설정
HOST = 'localhost'
PORT = 5000
MODEL_SIZE = int(os.environ.get("MODEL_SIZE", 1000))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.01))

class ParameterServer:
    def __init__(self, expected_workers):
        self.expected_workers = expected_workers
        self.weights = np.random.randn(MODEL_SIZE).astype(np.float32)
        
        self.lock = threading.Lock()
        self.barrier_lock = threading.Lock()
        
        self.collected_grads = {}
        self.current_round = 0
        self.connected_workers = 0
        
        self.shutdown_flag = False
        
    def handle_worker(self, conn, addr, worker_id):
        print(f"[Master] Worker-{worker_id} 연결됨 ({addr[0]}:{addr[1]})")
        
        with self.lock:
            self.connected_workers += 1
            if self.connected_workers == self.expected_workers:
                print(f"[Master] ✅ 모든 Worker({self.expected_workers}개) 가 연결되었습니다. 학습 시작!")
        
        try:
            while not self.shutdown_flag:
                # 1. 현재 가중치 전송
                with self.lock:
                    data = pickle.dumps(self.weights)
                length = len(data)
                conn.sendall(struct.pack('!I', length) + data)
                
                # 2. 기울기 수신
                raw_length = self._recv_all(conn, 4)
                if not raw_length:
                    break
                msg_len = struct.unpack('!I', raw_length)[0]
                grad_data = self._recv_all(conn, msg_len)
                gradient = pickle.loads(grad_data)
                
                # 3. Barrier 동기화: 모든 워커로부터 기울기 수집
                with self.barrier_lock:
                    self.collected_grads[worker_id] = gradient
                    
                    if len(self.collected_grads) >= self.expected_workers:
                        # 평균 계산 (All-Reduce)
                        avg_grad = np.mean([g for g in self.collected_grads.values()], axis=0)
                        
                        # 가중치 업데이트 (SGD)
                        self.weights -= LEARNING_RATE * avg_grad
                        
                        self.current_round += 1
                        print(f"[Master] Round {self.current_round}: {len(self.collected_grads)}개 Worker 로부터 기울기 집계 완료, 가중치 업데이트")
                        
                        # 초기화
                        self.collected_grads = {}
                
        except Exception as e:
            print(f"[Master] Worker-{worker_id} 오류: {e}")
        finally:
            conn.close()
            with self.lock:
                self.connected_workers -= 1
            print(f"[Master] Worker-{worker_id} 연결 종료")

    def _recv_all(self, conn, length):
        """지정된 길이만큼 데이터를 모두 수신"""
        data = b''
        while len(data) < length:
            packet = conn.recv(length - len(data))
            if not packet:
                return None
            data += packet
        return data

    def start(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen()
        
        print("=" * 50)
        print("🚀 분산 학습 파라미터 서버 (Master)")
        print("=" * 50)
        print(f"  - 바인딩: {HOST}:{PORT}")
        print(f"  - 모델 크기: {MODEL_SIZE} 차원")
        print(f"  - 학습률: {LEARNING_RATE}")
        print(f"  - 예상 Worker 수: {self.expected_workers}")
        print("=" * 50)
        
        worker_id_counter = 0
        
        try:
            while True:
                conn, addr = server_socket.accept()
                thread = threading.Thread(target=self.handle_worker, args=(conn, addr, worker_id_counter))
                thread.daemon = True
                thread.start()
                worker_id_counter += 1
        except KeyboardInterrupt:
            print("\n[Master] 서버를 종료합니다...")
        finally:
            self.shutdown_flag = True
            server_socket.close()

if __name__ == "__main__":
    num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("NUM_WORKERS", 3))
    print(f"[Master] 설정: Workers={num_workers}")
    server = ParameterServer(expected_workers=num_workers)
    server.start()
