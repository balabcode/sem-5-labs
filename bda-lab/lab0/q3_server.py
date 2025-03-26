# write a python program to read the - client will send a file to a server, which will count the number of words in it and send that back to client.

import socket
import threading

HOST = "127.0.0.1"
PORT = 12345

connections = []

def handle_client(conn, addr):
    while True:
        try:
            data = conn.recv(1024).decode('utf-8')
            if not data:
                break
            print(f"[*] Received from {addr}: {data}")
            for connection in connections:
                if connection != conn:
                    try:
                        connection.sendall(data.encode('utf-8'))
                    except socket.error:
                        pass
        except ConnectionResetError:
            print(f"[!] Client {addr} disconnected")
            connections.remove(conn)
            conn.close()
            break

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(5)
    print(f"[*] Listening on {HOST}:{PORT}")

    while True:
        conn, addr = s.accept()
        print(f"[*] Accepted connection from {addr}")
        connections.append(conn)
        client_thread = threading.Thread(target=handle_client, args=(conn, addr))
        client_thread.start()
