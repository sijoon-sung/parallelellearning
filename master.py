import socket
import pickle
import torch
import threading
from collections import OrderedDict

class ParameterServer:
    def __init__(self, host='0.0.0.0', port=9999, num_workers=2, lr=0.01):
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.lr = lr
        self.workers = []          # (socket, worker_id)
        self.global_weights = {'w': torch.tensor([0.0]), 'b': torch.tensor([0.0])}
        self.epoch = 0
        self.max_epochs = 10

    def start(self):
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((self.host, self.port))
        server_sock.listen(self.num_workers)
        print(f"[Master] Listening on {self.host}:{self.port}, waiting for {self.num_workers} workers...")

        # 1. 모든 Worker 연결 수락
        for i in range(self.num_workers):
            client_sock, addr = server_sock.accept()
            # Worker가 먼저 자신의 ID를 보냄
            data = self.recv_obj(client_sock)
            worker_id = data['worker_id']
            self.workers.append((client_sock, worker_id))
            print(f"[Master] Connected worker {worker_id} from {addr}")

        print("[Master] All workers connected. Starting training...\n")

        # 2. 에폭 반복
        for epoch in range(self.max_epochs):
            self.epoch = epoch
            print(f"[Master] Epoch {epoch+1}/{self.max_epochs}")

            # 모든 Worker 로부터 기울기 수집
            gradients = []
            for sock, wid in self.workers:
                grad_data = self.recv_obj(sock)
                gradients.append(grad_data['gradients'])
                print(f"[Master] Received gradients from worker {wid}")

            # 기울기 평균
            avg_grad = self.average_gradients(gradients)

            # Global 가중치 업데이트 (SGD)
            self.global_weights['w'] -= self.lr * avg_grad['w']
            self.global_weights['b'] -= self.lr * avg_grad['b']

            # 업데이트된 가중치를 모든 Worker에게 브로드캐스트
            for sock, wid in self.workers:
                self.send_obj(sock, {'weights': self.global_weights})
                print(f"[Master] Sent new weights to worker {wid}")

        # 종료
        for sock, _ in self.workers:
            self.send_obj(sock, {'exit': True})
            sock.close()
        server_sock.close()
        print("[Master] Training finished.")

    def average_gradients(self, grad_list):
        """grad_list: list of dicts {'w': tensor, 'b': tensor}"""
        avg_w = sum(g['w'] for g in grad_list) / len(grad_list)
        avg_b = sum(g['b'] for g in grad_list) / len(grad_list)
        return {'w': avg_w, 'b': avg_b}

    def recv_obj(self, sock):
        """소켓으로부터 길이+pickle 데이터를 수신하여 객체로 복원"""
        raw_len = sock.recv(4)
        if not raw_len:
            return None
        obj_len = int.from_bytes(raw_len, 'big')
        data = b''
        while len(data) < obj_len:
            chunk = sock.recv(min(4096, obj_len - len(data)))
            if not chunk:
                break
            data += chunk
        return pickle.loads(data)

    def send_obj(self, sock, obj):
        """객체를 pickle + 길이 헤더로 전송"""
        data = pickle.dumps(obj)
        sock.sendall(len(data).to_bytes(4, 'big') + data)

if __name__ == '__main__':
    ps = ParameterServer(num_workers=2, lr=0.01)  # Worker 수에 맞게 조정
    ps.start()