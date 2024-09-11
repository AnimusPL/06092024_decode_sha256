import hashlib
import time
from itertools import combinations, islice
from datetime import datetime

# Wzorzec docelowego hasha
target_hash = "0a2551e134fca8fc124380498e7802f79576fac3291f855a6030225dd5839717"

# Funkcja do generowania hashów SHA-256 na CPU
def sha256_hash_cpu(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# Zmienna globalna do przechowywania bieżącej kombinacji
current_combination = [""] * 4

# Funkcja do sprawdzania pojedynczej kombinacji
def check_combination(combination):
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

# Funkcja do wyświetlania postępu
def print_progress(count):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    # Wyświetlanie aktualnej kombinacji i liczby przetworzonych kombinacji
    print(f"[{now}] Przetworzono {count} kombinacji, obecna kombinacja: {''.join(current_combination)}")

# Ścieżka do pliku ze słownikiem
input_file = 'lista_slow_reduced.txt'  # Upewnij się, że plik jest w tym samym katalogu lub podaj pełną ścieżkę

# Wczytanie słów ze słownika
words = load_words(input_file)

# Informacja o liczbie słów
print(f"Liczba słów w słowniku: {len(words)}")

if not words:
    print("Lista słów jest pusta. Sprawdź plik wejściowy.")
else:
    print("Rozpoczynam generowanie kombinacji...")

    count = 0  # Zmienna do liczenia liczby przetworzonych kombinacji
    batch_size = 500  # Ilość kombinacji w jednej partii

    # Przetwarzanie kombinacji w partiach
    combination_generator = combinations(words, 4)

    start_time = time.time()
    
    while True:
        batch = list(islice(combination_generator, batch_size))
        if not batch:
            break

        # Sprawdzanie każdej kombinacji
        for combo in batch:
            result = check_combination(combo)
            count += 1

            # Co minutę wypisuje postęp
            if time.time() - start_time >= 60:
                print_progress(count)
                start_time = time.time()

            # Sprawdź, czy znaleziono rozwiązanie
            if result is not None:
                print(f"Znaleziono hasło: {result}")
                break
        else:
            continue
        break
