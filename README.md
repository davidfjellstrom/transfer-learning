# Avfallsklassificering med Transfer Learning

Det här är vårt grupprojekt i kursen *Tillämpad AI, datautvinning, maskininlärning och deep learning* (YH-utbildning).

**Vad vi bygger:** ett AI-program som tittar på en bild av skräp och gissar vilken typ det är — **återvinningsbart**, **elektronik** eller **organiskt**. Vi undersöker också hur mycket träningsdata som faktiskt behövs för att det ska funka bra, jämfört med att träna en modell helt från noll.

**Gruppmedlemmar:** David Fjellström, Anton Hergefelt & Gabriella Cross

## Bakgrund

Om skräp sorteras fel kostar det pengar för återvinningsföretag. I värsta fall hamnar elektronik i fel soptunna, vilket kan skada både utrustning och miljö. Ett AI-program som känner igen avfallstyp på en bild kan hjälpa till att sortera rätt, eller dubbelkolla det en människa redan gjort.

Vi bygger en bildklassificerare (ett AI-program som sorterar bilder i kategorier) med en teknik som heter **transfer learning**. Det betyder att vi lånar en modell som redan är bra på att känna igen saker i bilder — den har tränats på miljontals bilder tidigare — och lär om den till vår uppgift, istället för att börja från noll. Vi jämför den mot en modell som tränas helt från grunden, för att se om det faktiskt gör skillnad. Resultatet presenteras muntligt som en berättelse: vad är problemet, vad är lösningen, och varför skulle ett företag bry sig?

## Projektmål och frågeställningar

Målet är att visa att transfer learning löser avfallssorteringsproblemet bättre, och med mindre data, än en modell tränad från grunden — och att räkna ut vad det är värt för ett företag i praktiken. Allt i notebooken ska svara på tre frågor, i den här ordningen (fråga 1 är viktigast):

1. **Fråga 1 — Hur mycket data behövs?** Hur få bilder klarar transfer learning sig med för att bli bra, jämfört med att träna samma typ av modell helt från noll? Det här är resultatet som bär presentationen.
2. **Fråga 2 — Hur mycket av modellen behöver ändras?** Måste vi träna om hela den lånade modellen, eller räcker det med bara de sista lagren (stegen i modellen)? Det här förklarar *varför* transfer learning funkar, inte bara *att* den gör det.
3. **Fråga 3 — Vad kostar ett misstag?** Vilka felgissningar är farligast eller dyrast för ett återvinningsföretag (t.ex. att tro att elektronik är organiskt avfall)? Och hur borde det påverka hur modellen används i praktiken?

Varje del av notebooken ska gå att koppla till en av de tre frågorna, så presentationen blir en sammanhängande berättelse istället för lösryckta delar.

### Experiment 1 (fråga 1): hur mycket data behövs

Vi har många bilder (26 527 st), så "vi fick 99% rätt" säger inte så mycket i sig själv — det säger inget om *hur* vi kom dit. Vi tränar två sorters modeller (transfer learning och en modell från noll) på växande mängder bilder (t.ex. 50, 100, 500, 2000 och alla bilder per kategori) och ritar en kurva: antal bilder på x-axeln, hur ofta modellen gissar rätt på y-axeln. Den kurvan blir presentationens huvudbild.

### Experiment 2 (fråga 2): hur mycket av modellen behöver ändras

Med den bästa datamängden från experiment 1 tränar vi samma modell (ResNet50V2) flera gånger, men tinar upp (gör tränbara) olika många lager varje gång — bara vårt eget huvud, sista ~10 lagren, sista ~30 lagren, eller hela modellen. Vi ritar en kurva över hur bra modellen blir beroende på hur mycket som tinas upp, och förklarar varför: tidiga lager har lärt sig generella saker (kanter, former) som funkar oavsett bildtyp, medan sena lager är mer specialiserade och behöver anpassas till just vår uppgift.

### Experiment 3 (fråga 3): vad kostar ett misstag

På den bästa modellen (från experiment 1 och 2): en confusion matrix (en tabell som visar exakt vilka kategorier modellen blandar ihop), en classification_report (mer detaljerade siffror per kategori), och en diskussion om vilket misstag som är dyrast för ett återvinningsföretag och hur det borde påverka hur modellen används.

## Dataset

[`fadlicr7/waste-classification-dataset`](https://www.kaggle.com/datasets/fadlicr7/waste-classification-dataset) från Kaggle.

- 26 527 märkta bilder (bilder där vi redan vet rätt svar) i tre kategorier: `Recyclable`, `Electronic`, `Organic`
- Licens: CC0 (fri att använda)
- `test/`-mappen (1 458 bilder) hör till en tävling hos den som skapade datasetet och saknar facit — vi använder **inte** den som vårt testset. Vi delar istället upp våra egna 26 527 märkta bilder själva.

### Ladda ner data

```bash
kaggle datasets download -d fadlicr7/waste-classification-dataset -p data/raw --unzip
```

Kräver ett Kaggle-konto kopplat till datorn (en fil som heter `~/.kaggle/kaggle.json`) — se [Kaggles egen guide](https://www.kaggle.com/docs/api) om det inte redan är gjort.

## Teknisk stack

- **Keras / TensorFlow** — ett bibliotek (en färdig verktygslåda) för att bygga och träna AI-modeller. Vi använder `ResNet50V2`, en färdigtränad bildmodell, som bas för transfer learning.
- En enkel egenbyggd CNN (en vanlig typ av bildmodell) helt utan förträning, som jämförelsepunkt ("scratch"-modellen)
- `tf.image.resize_with_pad()` — skalar alla bilder till samma storlek (224×224 pixlar) utan att förvränga dem
- scikit-learn för utvärdering (confusion matrix m.m.), matplotlib för diagram

Vi tränar i **Google Colab** (gratis, ger tillgång till en GPU som gör träningen mycket snabbare) — notebooken ska gå att köra där utan ändringar. All kod (datahantering, modeller, experiment, utvärdering) ligger direkt i notebooken, inte i separata Python-filer — enklare att öppna och köra rakt av i Colab.

## Projektstruktur

```
transfer-learning/
├── README.md
├── requirements.txt               # tensorflow, numpy, matplotlib, scikit-learn, pillow, kaggle
├── data/
│   ├── raw/                        # nedladdad data (gitignorad)
│   └── splits/                     # vår egen train/val/test-uppdelning (gitignorad)
├── notebooks/
│   └── transfer_learning_waste.ipynb   # all kod: datahantering, modeller, experiment, utvärdering
├── reports/
│   └── figures/                     # sparade diagram/bilder till presentationen
└── .gitignore
```

## Plan

### 1. Grundstruktur och miljö
- [x] Ladda ner dataset till `data/raw`
- [x] Sätt upp git-repo, så vi kan jobba i varsin branch
- [x] `requirements.txt` och `.gitignore` (`data/`, `*.h5`, checkpoints, `.env`)
- [x] Skapa mappstrukturen ovan

### 2. Databehandling
- [x] Kolla hur många bilder vi har per kategori, och titta på exempelbilder (notebook, avsnitt 2, se `reports/figures/`)
- [x] Dela upp `train/`-bilderna själva i träning/validering/test (inte Kaggle-tävlingens omärkta `test/`)
- [x] Funktion som drar lika många bilder per kategori (balanserad delmängd), i valfri storlek

### 3. Modeller
- [x] Transfer learning-modell: `ResNet50V2` som bas + eget klassificeringshuvud (de sista lagren som gör den faktiska gissningen)
- [x] En enkel modell utan förträning ("scratch"), med jämförbar storlek, som jämförelsepunkt
- [x] Bestämde tvåstegs fine-tuning (frys allt → träna huvudet, tina upp sista lagren → träna om) som standard (notebook, avsnitt 3) — ett-stegsvarianten (lärarens exempel) finns kvar till frysdjups-experimentet

### 4. Experiment 1 — hur mycket data behövs (fråga 1)
- [x] `run_experiment1` / `plot_experiment1` körda (notebook avsnitt 4). Bäst TL-resultat: 92,5% med alla bilder, men redan 87,5% med bara 50 bilder/kategori — mot scratch-modellens 48,3% på samma lilla datamängd. Resultat i `reports/experiment1_results.json`, kurvan i `reports/figures/experiment1_data_needed.png`

### 5. Experiment 2 — hur mycket av modellen behöver ändras (fråga 2)
- [x] `run_experiment2` / `plot_experiment2` körda med `BEST_SUBSET_SIZE = 2773` (notebook avsnitt 5). Bäst resultat: **30 upptinade lager** (93,4%) — bättre än både bara huvudet (92,7%) och alla 190 lager (93,3%). Resultat i `reports/experiment2_results.json`, kurvan i `reports/figures/experiment2_unfrozen_layers.png`

### 6. Experiment 3 & utvärdering — vad kostar ett misstag (fråga 3)
- [x] `train_best_model` / `evaluate_best_model` körda med `BEST_UNFROZEN_LAYERS = 30` (notebook avsnitt 6). Slutmodell: 92% accuracy på testsetet. Electronic förväxlas nästan aldrig med övriga kategorier (recall 97%) — den vanligaste förväxlingen är Recyclable/Organic, ett billigare misstag. Confusion matrix i `reports/figures/confusion_matrix.png`, helhetsresultat i `reports/final_summary.json`
- [ ] Grad-CAM — hoppades över, prioriterades inte (inte kritiskt för presentationen)

### 7. Notebook och presentation
- [x] `notebooks/transfer_learning_waste.ipynb` klar rakt igenom, avsnitt 1–7 kodade, körda och med riktig output
- [x] Affärsnytta, begränsningar och svar på alla tre frågor skrivna i avsnitt 7
- [ ] Förbered den muntliga presentationen (~15 min) — bygg på kurvorna från experiment 1–2 och confusion matrix från experiment 3

## Kör lokalt

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
kaggle datasets download -d fadlicr7/waste-classification-dataset -p data/raw --unzip
jupyter notebook notebooks/transfer_learning_waste.ipynb
```

## Kör i Google Colab

Öppna `notebooks/transfer_learning_waste.ipynb` i Colab, sätt på GPU (Runtime → Change runtime type → GPU), och kör cellerna i ordning. Notebooken laddar ner och installerar det den behöver själv.

---

*Ursprunglig projektspecifikation: [claude_code_prompt_waste_classification_1.md](claude_code_prompt_waste_classification_1.md)*
