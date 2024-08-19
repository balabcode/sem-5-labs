import socket
import threading

HOST = "127.0.0.1"
PORT = 12345

def receive_messages(conn):
    while True:
        try:
            data = conn.recv(1024).decode('utf-8')
            if not data:
                break
            print(f"Received from server: {data}")
        except ConnectionResetError:
            print("[!] Disconnected from server")
            conn.close()
            break

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conn:
    conn.connect((HOST, PORT))
    receive_thread = threading.Thread(target=receive_messages, args=(conn,))
    receive_thread.start()

    while True:
        message = input("Enter message to send (or 'exit' to quit): ")
        if message.lower() == 'exit':
            break
        conn.sendall(message.encode('utf-8'))
