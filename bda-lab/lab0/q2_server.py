# write a python program to read the - client will send a file to a server, which will count the number of words in it and send that back to client.

import socket

HOST = "127.0.0.1"
PORT = 65433

def read_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.replace("-", " ")
    return text

def count_words(text):
    return len(text.split()), text.split()

def reverse_freq(text_arr):
    words = {}
    for word in text_arr:
        if word in words:
            words[word] += 1
        else:
            words[word] = 1
    words_sorted = ""
    for word, freq in sorted(words.items(), key = lambda item: item[1], reverse=True):
        words_sorted += f"{word} : {freq}\n"
    return words_sorted

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            data = conn.recv(1024)
            if not data:
                break
            text = read_file(data)
            num_words, word_arr = count_words(text)
            conn.sendall(bytes(reverse_freq(word_arr), encoding="utf-8"))