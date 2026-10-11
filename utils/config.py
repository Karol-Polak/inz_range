"""
Nazwane stałe (progi, rozmiary jąder morfologicznych, ułamki powierzchni
itd.) używane przez detekcję tarczy i trafień
(services/target_detection.py, services/hit_detection.py).

Wydzielone tutaj z dwóch powodów:
  - łatwiej je opisać / uzasadnić w pracy inżynierskiej,
  - można je dostrajać bez grzebania w logice detekcji.

To jest tylko przeniesienie istniejących wartości literalnych pod nazwy -
same wartości i zachowanie detekcji się NIE zmieniły (patrz
tests/test_detection_config_refactor_equivalence.py).
"""

# ---------------------------------------------------------------------------
# target_detection.py - wykrywanie dużego ciemnego obszaru (sylwetka/tarcza
# nieregularna), próbowane jako pierwsze
# ---------------------------------------------------------------------------
FOREGROUND_CLOSE_KERNEL_SIZE = (13, 13)  # domyka drobne przerwy w masce ciemnego obszaru
FOREGROUND_CLOSE_ITERATIONS = 2
FOREGROUND_MIN_AREA_FRACTION = 0.10      # obszar musi zajmować >=10% zdjęcia, by uznać go za tarczę
FOREGROUND_MIN_FILL_RATIO = 0.20         # area / (bbox area) - odrzuca bardzo "dziurawe"/postrzępione kontury
FOREGROUND_CONFIDENCE = 0.75             # pewność zwracana dla tego typu detekcji (heurystyka, nie pomiar)

# ---------------------------------------------------------------------------
# target_detection.py - wykrywanie okręgu (tarcza kołowa) metodą Hougha,
# używane gdy powyższa heurystyka nic nie znajdzie
# ---------------------------------------------------------------------------
CIRCULAR_BLUR_KERNEL_SIZE = (9, 9)   # wygładzenie przed Houghem, redukuje szum krawędzi
CIRCULAR_BLUR_SIGMA = 1.5
HOUGH_DP = 1.2                       # rozdzielczość akumulatora Hougha (1/dp rozmiaru obrazu)
HOUGH_MIN_DIST_PX = 100              # min. odległość między środkami wykrytych okręgów
HOUGH_PARAM1 = 100                   # próg wyższy dla detektora krawędzi Canny użytego wewnętrznie
HOUGH_PARAM2 = 20                    # próg akumulatora - im niższy, tym więcej (fałszywych) okręgów
HOUGH_MIN_RADIUS_FRACTION = 0.20     # min. promień jako ułamek krótszego wymiaru obrazu
HOUGH_MAX_RADIUS_FRACTION = 0.49     # maks. promień jako ułamek krótszego wymiaru obrazu
CIRCULAR_CENTER_WINDOW_MIN_FRACTION = 0.15  # środek okręgu musi leżeć w tym oknie obrazu (x i y)
CIRCULAR_CENTER_WINDOW_MAX_FRACTION = 0.85
CIRCULAR_CENTER_DISTANCE_PENALTY = 0.35     # waga karania odległości środka okręgu od środka zdjęcia przy wyborze najlepszego kandydata
CIRCULAR_CONFIDENCE = 1.0

# ---------------------------------------------------------------------------
# hit_detection.py - maska obszaru tarczy dla tarcz kołowych (gdzie mogą
# być trafienia)
# ---------------------------------------------------------------------------
CENTRAL_MASK_DARK_THRESHOLD = 245            # piksel uznany za "ciemny" (część tarczy), jeśli < tej wartości
CENTRAL_MASK_SATURATION_THRESHOLD = 40       # albo wystarczająco nasycony kolorystycznie (HSV S)
CENTRAL_MASK_CLOSE_KERNEL_SIZE = (17, 17)
CENTRAL_MASK_CLOSE_ITERATIONS = 1
CENTRAL_MASK_MIN_COMPONENT_AREA_FRACTION = 0.0004  # odrzuca drobne, przypadkowe plamki
CENTRAL_MASK_CENTER_LIMIT_FRACTION = 0.62    # jak daleko od środka tarczy może leżeć składowa maski; używane też jako promień koła awaryjnego, gdy maska wyjdzie pusta

# ---------------------------------------------------------------------------
# hit_detection.py - maska obszaru tarczy dla tarcz "foreground" (sylwetek)
# ---------------------------------------------------------------------------
FOREGROUND_MASK_CLOSE_KERNEL_SIZE = (15, 15)
FOREGROUND_MASK_CLOSE_ITERATIONS = 2
FOREGROUND_MASK_MIN_CONTOUR_AREA_FRACTION = 0.001
FOREGROUND_MASK_MIN_FILL_RATIO = 0.08
FOREGROUND_MASK_ASPECT_RATIO_MIN = 0.15      # poza tym zakresem kontur jest zbyt "wąski"...
FOREGROUND_MASK_ASPECT_RATIO_MAX = 6.5
FOREGROUND_MASK_FALLBACK_FILL_RATIO = 0.22   # ...chyba że fill_ratio jest wystarczająco wysoki, by mu to wybaczyć
FOREGROUND_MASK_DISTANCE_FRACTION = 1.05     # odrzuca kontury dalej niż promień tarczy * ten współczynnik

# ---------------------------------------------------------------------------
# hit_detection.py - wykrywanie jasnych kandydatów na trafienia (przestrzeliny)
# ---------------------------------------------------------------------------
BRIGHT_TOPHAT_KERNEL_SIZE = (23, 23)  # top-hat wyłapuje lokalne jasne detale niezależne od globalnego tła
BRIGHT_ABSOLUTE_THRESHOLD = 205       # piksel jasny w sposób absolutny (niezależnie od lokalnego kontrastu)
BRIGHT_TARGET_NEIGHBORHOOD_DILATE_KERNEL_SIZE = (9, 9)  # margines wokół maski tarczy, w którym liczą się kandydaci
BRIGHT_TARGET_NEIGHBORHOOD_DILATE_ITERATIONS = 1
BRIGHT_OPEN_KERNEL_SIZE = (3, 3)      # usuwa pojedyncze piksele szumu z maski kandydatów
BRIGHT_OPEN_ITERATIONS = 1

# ---------------------------------------------------------------------------
# hit_detection.py - rozdzielanie stykających się/nakładających trafień
# metodą watershed
# ---------------------------------------------------------------------------
SPLIT_MIN_AREA = 160           # poniżej tego obszaru nie próbujemy dzielić (zbyt małe, by to miało sens)
SPLIT_MIN_DIMENSION = 18       # ...podobnie jeśli żaden wymiar bbox nie jest większy niż to
SPLIT_PEAK_THRESHOLD_FRACTION = 0.55  # próg na mapie odległości (0.55 * maksimum) wyznaczający "szczyty" = środki dziur
SPLIT_MIN_DISTINCT_PEAKS = 2    # cv2.connectedComponents liczy też tło jako etykietę 0, więc potrzeba > tej liczby etykiet, by uznać, że są >=2 odrębne szczyty
SPLIT_SEPARATED_DILATE_KERNEL_SIZE = (3, 3)  # delikatnie poszerza każdy odseparowany kawałek z powrotem
SPLIT_SEPARATED_DILATE_ITERATIONS = 1

# ---------------------------------------------------------------------------
# hit_detection.py - dopuszczalny zakres powierzchni kandydata na trafienie
# (jako ułamek powierzchni całego zdjęcia, z dolnym/górnym limitem w pikselach)
# ---------------------------------------------------------------------------
CANDIDATE_MIN_AREA_FLOOR = 18
CANDIDATE_MIN_AREA_FRACTION = 0.000006
CANDIDATE_MAX_AREA_FLOOR = 900
CANDIDATE_MAX_AREA_FRACTION = 0.0016

# ---------------------------------------------------------------------------
# hit_detection.py - filtry kształtu stosowane do każdego kandydata w
# głównej pętli detect_hit()
# ---------------------------------------------------------------------------
HIT_ASPECT_RATIO_MIN = 0.55    # odrzuca kandydatów zbyt odbiegających od kwadratowego bbox...
HIT_ASPECT_RATIO_MAX = 1.8     # ...w obie strony
HIT_MIN_FILL_RATIO = 0.38      # area / (bbox area) - odrzuca kształty zbyt "rozlane" wewnątrz bbox
HIT_MIN_CIRCULARITY = 0.65     # 4*pi*area/perimeter^2 - 1.0 to idealne kółko
HIT_CONFIDENCE_MAX = 1.0       # pewność trafienia = min(HIT_CONFIDENCE_MAX, circularity)

# ---------------------------------------------------------------------------
# hit_detection.py - czy kandydat leży wystarczająco "w obrębie" tarczy
# ---------------------------------------------------------------------------
INSIDE_TARGET_RADIUS_FLOOR = 7        # minimalny promień sprawdzanego sąsiedztwa (dla bardzo małych kandydatów)
INSIDE_TARGET_RADIUS_MULTIPLIER = 1.8  # promień sąsiedztwa = większy wymiar bbox kandydata * ten współczynnik
INSIDE_TARGET_MIN_RATIO = 0.35        # jaka część tego sąsiedztwa musi pokrywać się z maską tarczy

# ---------------------------------------------------------------------------
# hit_detection.py - odrzucanie fragmentów wydrukowanych cyfr/oznaczeń
# (np. numerów pól punktacji) błędnie wyglądających jak trafienie
# ---------------------------------------------------------------------------
DIGIT_ESCAPE_CIRCULARITY = 0.82   # kandydat wystarczająco okrągły/wypełniony/duży jest ZAWSZE
DIGIT_ESCAPE_FILL_RATIO = 0.55    # uznawany za prawdziwe trafienie, bez sprawdzania sąsiedztwa
DIGIT_ESCAPE_MIN_AREA = 45        # (te trzy warunki muszą być spełnione razem)
DIGIT_PADDING_FLOOR = 4           # margines wokół kandydata przeszukiwany w poszukiwaniu pobliskich "kresek"
DIGIT_PADDING_MULTIPLIER = 0.75   # margines = większy wymiar bbox kandydata * ten współczynnik
DIGIT_BRIGHT_RANGE_MIN = 150      # zakres jasności pikseli uznawanych za potencjalną kreskę druku
DIGIT_BRIGHT_RANGE_MAX = 255
DIGIT_STROKE_MIN_AREA = 8         # zbyt małe plamki w sąsiedztwie są ignorowane (szum)
DIGIT_STROKE_ELONGATED_MULTIPLIER = 1.5  # "kreska" = komponent wyraźnie szerszy/wyższy niż sam kandydat
DIGIT_STROKE_THIN_FILL_RATIO = 0.45      # ...albo wypełniający niewielką część własnego bbox (kreska, nie plamka)
DIGIT_MIN_NEARBY_STROKES = 2      # potrzeba >= tylu pobliskich "kresek", by odrzucić kandydata jako fragment druku

# ---------------------------------------------------------------------------
# hit_detection.py - usuwanie duplikatów tego samego trafienia
# ---------------------------------------------------------------------------
DEDUP_DISTANCE_PX = 12  # dwa wykrycia bliżej siebie niż to są tym samym trafieniem
