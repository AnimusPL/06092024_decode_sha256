import numpy as np
from numba import cuda
import hashlib
import threading
import time
from itertools import product, islice
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Wzorzec docelowego hasha
target_hash = bytes.fromhex("0a2551e134fca8fc124380498e7802f79576fac3291f855a6030225dd5839717")

# Funkcja CPU do generowania hashy SHA-256
def sha256_hash_cpu(text):
    return hashlib.sha256(text.encode('utf-8')).digest()

# Funkcja CUDA do porównywania hashy
@cuda.jit
def check_combination_gpu(combinations, target_hash, results):
    idx = cuda.grid(1)
    if idx < combinations.shape[0]:
        candidate = combinations[idx]
        # Porównanie hasha na GPU
        match = True
        for i in range(len(target_hash)):
            if candidate[i] != target_hash[i]:
                match = False
                break
        results[idx] = 1 if match else 0

# Funkcja do wczytywania listy słów z pliku
def load_words(file_path):
    words = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                word = line.strip()  # Usunięcie białych znaków
                if word:  # Sprawdzenie, czy linia nie jest pusta
                    words.append(word)
    except FileNotFoundError:
        print(f"Błąd: Plik '{file_path}' nie został znaleziony.")
    except Exception as e:
        print(f"Błąd podczas wczytywania słów: {e}")
    return words

# Funkcja do wyświetlania postępu co minutę
def progress_monitor():
    while not stop_event.is_set():
        time.sleep(60)  # Czekaj 60 sekund
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        # Wyświetlanie aktualnej kombinacji i liczby przetworzonych kombinacji
        if current_combination:
            print(f"[{now}] Przetworzono {count[0]} kombinacji, obecna kombinacja: {''.join(current_combination)}")

# Ścieżka do pliku ze słownikiem
input_file = 'lista_slow_filtered_filtered.txt'  # Upewnij się, że plik jest w tym samym katalogu lub podaj pełną ścieżkę

# Wczytanie słów ze słownika
words = load_words(input_file)

# Informacja o liczbie słów
print(f"Liczba słów w słowniku: {len(words)}")

if not words:
    print("Lista słów jest pusta. Sprawdź plik wejściowy.")
else:
    print("Rozpoczynam generowanie kombinacji...")

    # Utwórz zdarzenie do zatrzymania monitorowania postępu
    stop_event = threading.Event()
    count = [0]  # Używamy listy, aby mieć zmienną globalną dla wątków
    current_combination = ''  # Inicjalizuj jako pusty ciąg dla monitorowania

    # Uruchom wątek monitorujący
    monitor_thread = threading.Thread(target=progress_monitor)
    monitor_thread.start()

    # Parametry
    batch_size = 5000  # Ilość kombinacji w jednej partii
    max_workers = 12    # Liczba wątków odpowiadająca liczbie rdzeni w CPU
    max_length = 64     # Maksymalna długość dla tablicy

    # Tworzenie puli wątków
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Przetwarzanie kombinacji w partiach
        combination_generator = product(words, repeat=4)
        
        while True:
            batch = list(islice(combination_generator, batch_size))
            if not batch:
                break

            # Przekształć kombinacje do formatu odpowiedniego dla GPU
            batch_strings = [''.join(combo) for combo in batch]
            combinations_array = np.array([list(combo.encode('utf-8').ljust(max_length, b'\0')) for combo in batch_strings], dtype=np.uint8)
            results = np.zeros(len(combinations_array), dtype=np.int32)

            # Aktualizacja bieżącej kombinacji
            current_combination = batch[0] if batch else ''

            # Alokacja pamięci na GPU
            d_combinations = cuda.to_device(combinations_array)
            d_target_hash = cuda.to_device(np.array(list(target_hash), dtype=np.uint8))
            d_results = cuda.device_array_like(results)

            # Parametry CUDA
            threads_per_block = 1024  # Możesz spróbować zmienić na 256, jeśli wystąpią problemy
            blocks_per_grid = min((len(combinations_array) + (threads_per_block - 1)) // threads_per_block, 8192)

            # Wywołanie funkcji CUDA
            check_combination_gpu[blocks_per_grid, threads_per_block](d_combinations, d_target_hash, d_results)

            # Kopiowanie wyników z powrotem na CPU
            results = d_results.copy_to_host()

            # Zliczanie kombinacji i sprawdzanie wyników
            count[0] += len(batch)
            for i, result in enumerate(results):
                if result == 1:
                    print(f"Znaleziono hasło: {batch[i]}")
                    stop_event.set()  # Zatrzymaj monitorowanie postępu
                    break
            else:
                continue
            break
    
    # Zatrzymaj monitorowanie postępu
    stop_event.set()
    monitor_thread.join()  # Poczekaj na zakończenie wątku monitorującego
