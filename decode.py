import hashlib
import threading
import time
from itertools import product, islice
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Wzorzec docelowego hasha
target_hash = "0a2551e134fca8fc124380498e7802f79576fac3291f855a6030225dd5839717"

# Funkcja do generowania hashów SHA-256 na CPU
def sha256_hash_cpu(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# Zmienna globalna do przechowywania bieżącej kombinacji
current_combination = [""] * 4

# Funkcja do sprawdzania pojedynczej kombinacji na GPU
def check_combination_gpu(combination):
    global current_combination
    current_combination = combination  # Zaktualizuj bieżącą kombinację
    candidate = ''.join(combination)
    if sha256_hash_cpu(candidate) == target_hash:
        return candidate
    return None

# Wczytanie listy słów z pliku .txt
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

    # Uruchom wątek monitorujący
    monitor_thread = threading.Thread(target=progress_monitor)
    monitor_thread.start()

    # Parametry
    batch_size = 5000  # Ilość kombinacji w jednej partii
    max_workers = 12    # Liczba wątków odpowiadająca liczbie rdzeni w CPU

    # Tworzenie puli wątków
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Przetwarzanie kombinacji w partiach
        combination_generator = product(words, repeat=4)
        
        while True:
            batch = list(islice(combination_generator, batch_size))
            if not batch:
                break

            # Sprawdzanie każdej kombinacji w osobnym wątku
            results = [check_combination_gpu(combo) for combo in batch]
            count[0] += len(batch)

            # Sprawdź, czy znaleziono rozwiązanie
            for result in results:
                if result is not None:
                    print(f"Znaleziono hasło: {result}")
                    stop_event.set()  # Zatrzymaj monitorowanie postępu
                    break
            else:
                continue
            break
    
    # Zatrzymaj monitorowanie postępu
    stop_event.set()
    monitor_thread.join()  # Poczekaj na zakończenie wątku monitorującego
