# Definiujemy litery, na które muszą się zaczynać wyrazy
allowed_letters = {'m', 'o', 'p', 't', 'i', 'r', 'd'}

# Wczytujemy wyrazy z pliku
with open('lista_slow_filtered.txt', 'r', encoding='utf-8') as infile:
    words = infile.readlines()

# Filtrujemy wyrazy
filtered_words = [word.strip() for word in words if word.strip().lower()[0] in allowed_letters]

# Zapisujemy przefiltrowane wyrazy do nowego pliku
with open('lista_slow_filtered_filtered.txt', 'w', encoding='utf-8') as outfile:
    for word in filtered_words:
        outfile.write(f"{word}\n")

print("Filtracja zakończona. Przefiltrowane wyrazy zapisano do 'lista_slow_filtered_filtered.txt'.")
