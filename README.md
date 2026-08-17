# Avfallsklassificering med Transfer Learning

Grupprojekt i kursen *Tillämpad AI, datautvinning, maskininlärning och deep learning* (YH-utbildning). Vi bygger en bildklassificerare som sorterar avfall i tre kategorier — **Recyclable**, **Electronic**, **Organic** — och undersöker hur mycket träningsdata transfer learning faktiskt behöver jämfört med en modell tränad från scratch.

**Gruppmedlemmar:** David Fjellström, Anton Hergefelt

## Bakgrund

Felsorterat avfall kostar återvinningsanläggningar pengar, i värsta fall i form av elektronik som hamnar i fel ström och skadar utrustning eller miljö. En modell som automatiskt känner igen avfallstyp från bild kan stötta manuell sortering eller styra en sorteringslinje.

Vi tränar en bildklassificerare via transfer learning på en förtränad ImageNet-modell, och jämför den mot en modell tränad helt från scratch. Resultatet presenteras som en databerättelse: problemet, lösningen och varför den är värd något för ett företag.

## Projektmål och frågeställningar

Målet är att visa att transfer learning löser avfallssorteringsproblemet bättre och med mindre data än en modell tränad från grunden, och att räkna ut vad det är värt för ett företag i praktiken. Allt i notebooken byggs för att svara på tre frågor, i prioritetsordning:

1. **Huvudfråga — data-effektivitet:** hur mycket träningsdata krävs för att transfer learning ska bli bra, jämfört med att träna samma typ av modell från grunden? Detta är resultatet som bär presentationen.
2. **Fördjupning — frysdjup:** hur mycket av den förtränade basmodellen behöver egentligen tinas upp och tränas om för att prestandan ska bli bra? Förklarar *varför* transfer learning fungerar, inte bara *att* den gör det.
3. **Affärsram — felkostnad:** vilka felklassificeringar är farligast eller dyrast för en återvinningsaktör (t.ex. Electronic som hamnar i Organic-strömmen), och hur bör det påverka hur modellen används i praktiken (tröskel, manuell kontroll av osäkra fall)?

Varje sektion i notebooken ska gå att koppla tillbaka till en av dessa tre frågor, så att presentationen blir en sammanhängande berättelse.

### Experiment 1 (huvudfråga): data-effektivitet

Datasetet är stort (26 527 bilder), vilket gör "vi fick 99% accuracy" till en ointressant story i sig. Vi tränar två modelltyper på växande delmängder av träningsdata (t.ex. 50, 100, 500, 2000 och alla bilder per klass, balanserat) och plottar val-accuracy mot antal träningsbilder. Den kurvan — transfer learning vs. scratch — blir presentationens huvudbild.

### Experiment 2 (fördjupning): frysdjup

Med bästa datamängden från experiment 1, tränar vi samma ResNet50V2-arkitektur flera gånger med olika mängd upptinade lager (endast eget huvud → sista ~10 lagren → sista ~30 lagren → hela basen), och plottar val-accuracy mot antal upptinade lager. Kopplas i markdown till varför man fryser lager överhuvudtaget: tidiga lager har redan lärt sig generella mönster (kanter, texturer), sena lager behöver anpassas till vår uppgift.

### Experiment 3 (affärsram): felkostnad

På den slutliga bästa modellen (experiment 1 + 2 kombinerat): confusion matrix, `classification_report`, och en diskussion om vilken felklassificering som är dyrast för ett återvinningsföretag och hur det bör påverka beslutströsklar eller manuell kontroll av osäkra fall.

## Dataset

[`fadlicr7/waste-classification-dataset`](https://www.kaggle.com/datasets/fadlicr7/waste-classification-dataset) från Kaggle.

- 26 527 märkta bilder i tre klasser: `Recyclable`, `Electronic`, `Organic`
- Licens: CC0
- `test/`-mappen (1 458 bilder) är **omärkt** tävlingsdata från upphovspersonens egen tävling (BDC Satria Data 2026) och används **inte** som vårt testset — vi skär ut ett eget train/val/test-split ur de 26 527 märkta bilderna i `train/`

### Ladda ner data

```bash
kaggle datasets download -d fadlicr7/waste-classification-dataset -p data/raw --unzip
```

Kräver Kaggle-autentisering (`~/.kaggle/kaggle.json`) — se [Kaggle API-dokumentationen](https://www.kaggle.com/docs/api) om den inte redan är konfigurerad.

## Teknisk stack

- **Keras / TensorFlow**, med `ResNet50V2` (`weights='imagenet'`) som basmodell för transfer learning
- En enkel CNN utan förtränade vikter som jämförelsepunkt ("scratch"-modellen)
- `tf.image.resize_with_pad()` för att skala bilder till 224×224 utan att förvränga proportioner
- scikit-learn för utvärdering (confusion matrix, classification report), matplotlib för plots

Träning körs primärt i **Google Colab** (gratis GPU) — notebooken ska gå att köra där utan ändringar.

## Projektstruktur

```
transfer-learning/
├── README.md
├── requirements.txt               # tensorflow, numpy, matplotlib, scikit-learn, pillow, kaggle
├── data/
│   ├── raw/                        # nedladdad data (gitignorad)
│   └── splits/                     # eget train/val/test (gitignorad)
├── notebooks/
│   └── transfer_learning_waste.ipynb
├── src/
│   ├── data_prep.py                 # train/val/test-split, delmängder
│   ├── models.py                    # TL-modell, scratch-CNN, varianter med olika frysdjup
│   └── evaluate.py                  # confusion matrix, classification report, Grad-CAM
├── reports/
│   └── figures/                     # sparade plots för presentationen
└── .gitignore
```

## Plan

### 1. Grundstruktur och miljö
- [x] Ladda ner dataset till `data/raw`
- [x] Sätt upp git-repo med branches för parallellt arbete
- [x] `requirements.txt` och `.gitignore` (`data/`, `*.h5`, checkpoints, `.env`)
- [x] Skapa mappstrukturen ovan

### 2. Databehandling
- [x] Utforska klassbalans och exempelbilder per klass (`src/data_prep.py`, se `reports/figures/`)
- [x] Bygga eget train/val/test-split ur `train/` (INTE Kaggle-tävlingens `test/`)
- [x] Funktion för att dra balanserade delmängder av given storlek per klass

### 3. Modeller
- [ ] Transfer learning-modell: `ResNet50V2`-bas + eget klassificeringshuvud (GAP → Dense(128) → BatchNorm → Dropout → Dense(3, softmax))
- [ ] Scratch-CNN med jämförbar huvudarkitektur, inga förtränade vikter
- [ ] Bestäm ett- eller tvåstegs fine-tuning för TL-modellen och motivera valet

### 4. Experiment 1 — data-effektivitet (huvudfråga)
- [ ] Träna TL och scratch på varje delmängdsstorlek
- [ ] Logga val-accuracy/val-loss per (delmängdsstorlek × modelltyp)
- [ ] Plotta data-effektivitetskurvan (huvudbilden för presentationen)

### 5. Experiment 2 — frysdjup (fördjupning)
- [ ] Träna ResNet50V2 med varierande antal upptinade lager (0 / ~10 / ~30 / alla)
- [ ] Logga val-accuracy per frysdjup, med bästa datamängden från experiment 1
- [ ] Plotta frysdjup-kurvan och motivera i markdown varför tidiga vs. sena lager beter sig olika

### 6. Experiment 3 & utvärdering — felkostnad (affärsram)
- [ ] Confusion matrix och `classification_report` för slutgiltig bästa modell
- [ ] Diskussion: vilken felklassificering är dyrast för ett återvinningsföretag, och vad det betyder för beslutströsklar
- [ ] Om tid finns: Grad-CAM-heatmaps per klass

### 7. Notebook och presentation
- [ ] Bygg `notebooks/transfer_learning_waste.ipynb` med markdown-avsnitt som dubblar som talmanus, strukturerad kring de tre frågeställningarna
- [ ] Sammanfatta affärsnytta: kostnad för felsortering, vad modellen ger en återvinningsaktör, begränsningar
- [ ] Förbered 15-minuters muntlig presentation

## Kör lokalt

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
kaggle datasets download -d fadlicr7/waste-classification-dataset -p data/raw --unzip
jupyter notebook notebooks/transfer_learning_waste.ipynb
```

## Kör i Google Colab

Öppna `notebooks/transfer_learning_waste.ipynb` i Colab, aktivera GPU-runtime (Runtime → Change runtime type → GPU), och kör cellerna i ordning. Notebooken hämtar och installerar det den behöver.

# Övrigt
Skapa 

---

*Ursprunglig projektspecifikation: [claude_code_prompt_waste_classification_1.md](claude_code_prompt_waste_classification_1.md)*
