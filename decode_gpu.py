import numpy as np
import threading
import time
from numba import cuda

# Stałe SHA-256
K = np.array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
], dtype=np.uint32)

# Funkcje pomocnicze SHA-256
@cuda.jit(device=True)
def rotate_right(value, shift):
    return ((value >> shift) | (value << (32 - shift))) & 0xffffffff

@cuda.jit(device=True)
def sha256_transform(chunk, hash_values):
    a, b, c, d, e, f, g, h = hash_values

    w = cuda.local.array(64, dtype=np.uint32)

    for i in range(16):
        w[i] = int.from_bytes(chunk[4 * i:4 * (i + 1)], 'big')

    for i in range(16, 64):
        s0 = rotate_right(w[i - 15], 7) ^ rotate_right(w[i - 15], 18) ^ (w[i - 15] >> 3)
        s1 = rotate_right(w[i - 2], 17) ^ rotate_right(w[i - 2], 19) ^ (w[i - 2] >> 10)
        w[i] = (w[i - 16] + s0 + w[i - 7] + s1) & 0xffffffff

    for i in range(64):
        S1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25)
        ch = (e & f) ^ ((~e) & g)
        temp1 = (h + S1 + ch + K[i] + w[i]) & 0xffffffff
        S0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        temp2 = (S0 + maj) & 0xffffffff

        h = g
        g = f
        f = e
        e = (d + temp1) & 0xffffffff
        d = c
        c = b
        b = a
        a = (temp1 + temp2) & 0xffffffff

    hash_values[0] = (hash_values[0] + a) & 0xffffffff
    hash_values[1] = (hash_values[1] + b) & 0xffffffff
    hash_values[2] = (hash_values[2] + c) & 0xffffffff
    hash_values[3] = (hash_values[3] + d) & 0xffffffff
    hash_values[4] = (hash_values[4] + e) & 0xffffffff
    hash_values[5] = (hash_values[5] + f) & 0xffffffff
    hash_values[6] = (hash_values[6] + g) & 0xffffffff
    hash_values[7] = (hash_values[7] + h) & 0xffffffff

@cuda.jit
def sha256_hash_gpu(text, target_hash, result, progress):
    idx = cuda.grid(1)
    if idx < text.shape[0]:
        hash_values = cuda.local.array(8, dtype=np.uint32)
        hash_values[0] = 0x6a09e667
        hash_values[1] = 0xbb67ae85
        hash_values[2] = 0x3c6ef372
        hash_values[3] = 0xa54ff53a
        hash_values[4] = 0x510e527f
        hash_values[5] = 0x9b05688c
        hash_values[6] = 0x1f83d9ab
        hash_values[7] = 0x5be0cd19

        candidate_bytes = text[idx]

        chunk = cuda.local.array(64, dtype=np.uint8)
        for i in range(len(candidate_bytes)):
            chunk[i] = candidate_bytes[i]
        for i in range(len(candidate_bytes), 64):
            chunk[i] = 0x00

        sha256_transform(chunk, hash_values)

        computed_hash = cuda.local.array(32, dtype=np.uint8)
        for i in range(8):
            start = i * 4
            for j in range(4):
                computed_hash[start + j] = (hash_values[i] >> (24 - j * 8)) & 0xff

        match = True
        for i in range(32):
            if computed_hash[i] != target_hash[i]:
                match = False
                break

        if match:
            result[0] = 1

        progress[idx] = 1

def monitor_progress(progress, text_array):
    start_time = time.time()
    while True:
        time.sleep(60)
        checked = np.sum(progress)
        print(f"Kombinacje sprawdzone: {checked}/{len(text_array)}")
        if checked > 0:
            last_checked_idx = np.where(progress == 1)[0][-1]
            print(f"Aktualnie sprawdzana kombinacja: {text_array[last_checked_idx]}")
        elapsed_time = time.time() - start_time
        print(f"Czas trwania: {elapsed_time / 60:.2f} minut")

def main():
    # Załadowanie słownika
    with open('slownik.txt', 'r') as file:
        slownik = [line.strip() for line in file]

    # Określenie maksymalnej długości słowa
    max_length = max(len(word) for word in slownik)

    # Przygotowanie danych do obliczeń na GPU
    text_array = np.zeros((len(slownik), max_length), dtype=np.uint8)
    for i, word in enumerate(slownik):
        text_array[i, :len(word)] = list(map(ord, word))

    target_hash = np.array([
        0x9c, 0xb8, 0x92, 0x0e, 0xc0, 0xd4, 0x5a, 0xa0,
        0x64, 0x05, 0xf8, 0xe2, 0x44, 0xb1, 0x70, 0x80,
        0x7b, 0x5d, 0xe2, 0x95, 0xd8, 0x9e, 0xe6, 0xd6,
        0x02, 0x85, 0x8c, 0x69, 0x8c, 0x08, 0x6e, 0x0a
    ], dtype=np.uint8)

    d_text_array = cuda.to_device(text_array)
    d_result = cuda.device_array(1, dtype=np.uint32)
    d_progress = cuda.device_array(text_array.shape[0], dtype=np.uint32)

    blocks_per_grid = (text_array.shape[0] + 255) // 256
    threads_per_block = 256

    progress_thread = threading.Thread(target=monitor_progress, args=(d_progress.copy_to_host(), text_array))
    progress_thread.start()

    sha256_hash_gpu[blocks_per_grid, threads_per_block](d_text_array, target_hash, d_result, d_progress)

    result = d_result.copy_to_host()
    if result[0] == 1:
        print("Znaleziono hasło!")
    else:
        print("Hasło nie zostało znalezione.")

if __name__ == "__main__":
    main()
