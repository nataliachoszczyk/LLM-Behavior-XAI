# Temat projektu: Różnice w cechach zachowania i profilach behawioralnych różnych modeli językowych.

Celem projektu jest analiza i porównanie zachowań wybranych modeli językowych (LLM) pod kątem ich stylu generowania odpowiedzi. Badanie skoncentruje się na ilościowych i jakościowych cechach tekstu, takich jak długość odpowiedzi, długość zdań, poziom formalności, bogactwo słownictwa oraz spójność odpowiedzi.

Dodatkowo analizowane będą:
- różnice w odpowiedziach na parafrazy tego samego zapytania,
- różnice wynikające z języka zapytania (polski vs angielski).

Projekt ma na celu identyfikację charakterystycznych „profili behawioralnych” modeli oraz określenie, na ile są one stabilne i powtarzalne.

# Autorzy projektu
- Dominika Boguszewska
- Natalia Choszczyk
- Aleksandra Samsel

# Harmonogram

### Tydzień 1 (23-29 marca)
- doprecyzowanie pytań badawczych i hipotez,
- wybór modeli do porównania oraz przygotowanie środowiska projektowego,
- przygotowanie wstępnej listy promptów i kategorii tematycznych.

### Tydzień 2 (30 marca - 5 kwietnia)
- zaprojektowanie finalnego zestawu promptów bazowych (PL/EN) oraz ich parafraz,
- ustalenie parametrów eksperymentu (temperatura, długość odpowiedzi, liczba powtórzeń),
- implementacja skryptu do automatycznego pobierania odpowiedzi modeli.

### Tydzień 3 (6-12 kwietnia)
- weryfikacja jakości danych i korekta pipeline,
- uruchomienie właściwego zbierania danych.

### Tydzień 4 (13-19 kwietnia)
- pełne uruchomienie eksperymentów dla wszystkich modeli i wariantów promptów.

### Tydzień 5 (20-26 kwietnia)
- przygotowanie modułu ekstrakcji cech ilościowych (długość odpowiedzi, długość zdań),
- przygotowanie modułu cech jakościowych (formalność, bogactwo słownictwa, spójność).

### Tydzień 6 (27 kwietnia - 3 maja)
- przeprowadzenie analiz,
- wstępna wizualizacja wyników i porównanie modeli.

### Tydzień 7 (4-10 maja)
- identyfikacja charakterystycznych wzorców stylu dla każdego modelu,
- stworzenie dashboardu i dopracowanie wizualizacji.

### Tydzień 8 (11-17 maja)
- finalna korekta dokumentacji i wyników,
- przygotowanie raportu końcowego.

### Tydzień 9 (18-21 maja)
- oddanie projektu.

# Bibliografia

# Planowany zakres eksperymentów
Eksperyment zostanie przeprowadzony w formie kontrolowanego porównania: te same zestawy zapytań (oryginały i parafrazy, w języku polskim i angielskim) będą kierowane do wszystkich wybranych modeli. Następnie odpowiedzi zostaną zapisane, przetworzone i ocenione według wspólnego zestawu metryk ilościowych i jakościowych, co pozwoli wyznaczyć profile behawioralne modeli oraz ocenić ich stabilność. 

- **Modele:** porównanie co najmniej 3 modeli LLM o różnych architekturach lub dostawcach.
- **Zestaw promptów:** 10-20 promptów bazowych w języku polskim i angielskim, z 2-3 parafrazami dla każdego promptu.
- **Procedura:** każdy model otrzyma identyczny zestaw zapytań (prompt bazowy, parafrazy, wersja PL i EN) przy tych samych parametrach generacji, aby porównanie było możliwie obiektywne.
- **Cechy ilościowe:** liczba znaków i słów, średnia długość zdania, liczba zdań, wskaźniki różnorodności leksykalnej (np. TTR).
- **Cechy jakościowe:** formalność stylu, spójność logiczna, konsekwencja tonu, czytelność.
- **Sposób oceny:** metryki ilościowe będą liczone automatycznie, a cechy jakościowe ocenione według wspólnej skali (np. 1-5) i jednolitych kryteriów dla wszystkich modeli.
- **Porównania główne:**
	- model A vs model B vs model C,
	- prompt bazowy vs jego parafrazy,
	- zapytania polskie vs angielskie.
- **Interpretacja wyników:** dla każdego modelu powstanie krótki profil behawioralny opisujący dominujący styl odpowiedzi oraz jego stabilność między wariantami pytań, wraz z uzasadnieniem, które cechy miały największy wpływ na ten profil.

# Planowana funkcjonalność programu
- automatyczne wysyłanie zestawu promptów do wybranych modeli i zapisywanie odpowiedzi,
- obsługa wariantów eksperymentu: język zapytania, parafrazy, wielokrotne uruchomienia,
- moduł ekstrakcji i obliczania metryk tekstowych (ilościowych i jakościowych),
- moduł porównawczy generujący zestawienia między modelami,
- prosty dashboard podsumowujący wyniki (metryki, porównania modeli, wpływ parafraz i języka),
- generowanie wykresów i tabel do raportu (np. rozkłady długości, porównania formalności),

# Planowany stack technologiczny

