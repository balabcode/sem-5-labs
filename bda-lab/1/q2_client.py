# write a python program to read the - client will send a file to a server, which will count the number of words in it and send that back to client.

import socket

HOST = "127.0.0.1"
PORT = 65433

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    s.sendall(b"/home/lplab/Documents/220962448_Balaji/text_for_q1.txt")
    data = s.recv(1024)

words_sorted = str(data, encoding="utf-8")

print(f"Received {words_sorted}")