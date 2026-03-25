# Temat projektu: Różnice w cechach zachowania i profilach behawioralnych różnych modeli językowych.

Celem projektu jest analiza i porównanie zachowań wybranych modeli językowych (LLM) pod kątem ich stylu generowania odpowiedzi. Badanie skoncentruje się na ilościowych i jakościowych cechach tekstu, takich jak długość odpowiedzi, długość zdań, poziom formalności, bogactwo słownictwa oraz spójność odpowiedzi.

Dodatkowo analizowane będą:
- różnice w odpowiedziach na parafrazy tego samego zapytania,
- różnice wynikające z języka zapytania (polski vs angielski).

Projekt ma na celu identyfikację charakterystycznych „profili behawioralnych” modeli oraz określenie, na ile są one stabilne i powtarzalne. Dodatkowo w projekcie zostanie zbudowany własny model wyjaśnialny, którego zadaniem będzie przewidywanie, do którego modelu LLM należy dana odpowiedź, na podstawie wyodrębnionych cech tekstowych.

# Potencjalne pytania badawcze

- Czy różne modele mają stabilny, rozpoznawalny profil stylu odpowiedzi?
- Czy parafraza tego samego promptu istotnie zmienia styl odpowiedzi modelu?
- Czy język promptu (PL vs EN) zmienia profil behawioralny modelu?
- Czy różnice między modelami są większe niż różnice między parafrazami w obrębie jednego modelu?

# Autorzy projektu

- Dominika Boguszewska
- Natalia Choszczyk
- Aleksandra Samsel

# Harmonogram

### Tydzień 1 (23-29 marca)

- Analiza literatury potrzebnej do realizacji projektu,
- Doprecyzowanie pytań badawczych i hipotez,
- Wybór modeli do porównania oraz przygotowanie środowiska projektowego,
- Przygotowanie wstępnej listy promptów i kategorii tematycznych.

### Tydzień 2 (30 marca - 5 kwietnia)

- Zaprojektowanie finalnego zestawu promptów bazowych (PL/EN) oraz ich parafraz,
- Ustalenie parametrów eksperymentu (temperatura, długość odpowiedzi, liczba powtórzeń),
- Implementacja skryptu do automatycznego pobierania odpowiedzi modeli.

### Tydzień 3 (6-12 kwietnia)

- Weryfikacja jakości danych i korekta pipeline,
- Uruchomienie właściwego zbierania danych.

### Tydzień 4 (13-19 kwietnia)

- Pełne uruchomienie eksperymentów dla wszystkich modeli i wariantów promptów.

### Tydzień 5 (20-26 kwietnia)

- Przygotowanie modułu ekstrakcji cech ilościowych (długość odpowiedzi, długość zdań),
- Przygotowanie modułu cech jakościowych (formalność, bogactwo słownictwa, spójność).

### Tydzień 6 (27 kwietnia - 3 maja)

- Budowa własnego modelu wyjaśnialnego przewidującego, do którego modelu LLM należy dana odpowiedź
- Analiza działania i interpretowalności modelu
- Interpretacja wyników XAI

### Tydzień 7 (4-10 maja)

- Identyfikacja charakterystycznych wzorców stylu dla każdego modelu,
- Stworzenie dashboardu i dopracowanie wizualizacji.

### Tydzień 8 (11-17 maja)

- Finalna korekta dokumentacji i wyników,
- Przygotowanie raportu końcowego.

### Tydzień 9 (18-21 maja)

- Oddanie projektu.

# Bibliografia

- *Han, Bin, Deuksin Kwon, and Jonathan Gratch. "Personality Expression Across Contexts: Linguistic and Behavioral Variation in LLM Agents." arXiv preprint arXiv:2602.01063 (2026).* - Badanie sprawdzające, jak te same prompty osobowościowe przekładają się na różne wyniki lingwistyczne i emocjonalne w zależności od kontekstu.
- *Truong, Kimberly, et al. "Persona-augmented benchmarking: Evaluating llms across diverse writing styles." Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. 2025.* - Badanie analizujące, jak zmiany stylu pisania wpływają na wyniki modeli.
- *Alhetelah, Bushra, and Irfan Ahmad. "Measuring LLMs’ Sensitivity to Paraphrased Opinion Prompts." The Proceedings for the 15th Workshop on Computational Approaches to Subjectivity, Sentiment Social Media Analysis (WASSA 2026). 2026.* - Badanie analizujące, jak modele takie jak GPT-4 czy Llama reagują na zmiany powierzchniowe w tekście (parafrazy) przy zachowaniu tego samego sensu.
- *Paleyes, Andrei, et al. "Prompt variability effects on LLM code generation." arXiv preprint arXiv:2506.10204 (2025).* - Zawiera metodologię badania „sprawiedliwości promptów” (prompt fairness), czyli czy różne sformułowania tego samego problemu dają drastycznie inne wyniki.
- *Guo, Yanzhu, Guokan Shang, and Chloé Clavel. "Benchmarking linguistic diversity of large language models." Transactions of the Association for Computational Linguistics 13 (2025): 1507-1526.* - Dostarcza gotowych ram do oceny różnorodności leksykalnej, syntaktycznej i semantycznej.


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
- **Interpretacja wyników:** dla każdego modelu powstanie krótki profil behawioralny opisujący dominujący styl odpowiedzi oraz jego stabilność między wariantami pytań. Dodatkowo zbudowany zostanie własny model wyjaśnialny, którego zadaniem będzie przewidywanie, do którego modelu należy dana odpowiedź, oraz wskazanie, które cechy miały największy wpływ na tę predykcję.

# Komponent XAI

Zostanie to wykonane na poziomie wyjaśniania różnic w zachowaniu modeli na podstawie obserwowanych cech odpowiedzi. Po wygenerowaniu odpowiedzi dla zadanych promptów, z każdej zostaną wyciągnięte cechy. Następnie zostanie zbudowany prosty model interpretowalny lub model pomocniczy klasyfikujący, który na podstawie tych cech będzie przewidywał:
- który model wygenerował odpowiedź,
- czy prompt był po polsku czy po angielsku,
- czy odpowiedź pochodzi z promptu bazowego czy parafrazy.
Następnie z wykorzystaniem SHAP, chcemy zobaczyć które cechy najbardziej odpowiadają za odróżnianie modeli. Dzięki temu projekt nie tylko pokaże, że modele różnią się profilem behawioralnym, ale także wskaże, jakie konkretne właściwości językowe odpowiadają za te różnice. Takie podejście pozwala odpowiedzieć nie tylko na pytanie, czy modele różnią się stylem, ale również: w jaki sposób się różnią i które właściwości tekstu są za to odpowiedzialne.

# Planowana funkcjonalność programu

- Automatyczne wysyłanie zestawu promptów do wybranych modeli i zapisywanie odpowiedzi,
- Obsługa wariantów eksperymentu: język zapytania, parafrazy, wielokrotne uruchomienia,
- Moduł ekstrakcji i obliczania metryk tekstowych (ilościowych i jakościowych),
- Moduł porównawczy generujący zestawienia między modelami oraz dane wejściowe do własnego modelu wyjaśnialnego,
- Prosty dashboard podsumowujący wyniki (metryki, porównania modeli, wpływ parafraz i języka),
- Generowanie wykresów i tabel do raportu (np. rozkłady długości, porównania formalności),

# Planowany stack technologiczny

- **Język programowania:** `Python 3.12`
- **Środowisko wirtualne:** `venv`
- **Autoformatter i linter:** `ruff`
- **Uruchomianie aplikacji:** `make`
- **Struktura projektu:** `cookiecutter`
- **Śledzenie eksperymentów:** Weights & Biases, MLflow
- **Analiza tekstu:** `nltk`, `textstat`, `sentencepiece`
- **Wizualizacja:** `matplotlib`, `seaborn`, `plotly`
- **XAI:** `shap`
- **Inne biblioteki:** `numpy`, `pandas`, `torch`, `scikit-learn`
- **Modele:** API modeli językowych, własny model wyjaśnialny
- **Środowisko:** Google Colab, Kaggle, lokalne
- **Repozytorium kodu:** GitHub
