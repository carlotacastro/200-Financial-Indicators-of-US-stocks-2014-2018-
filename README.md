# Pràctica Kaggle APC UAB 2021-22
### Nom: Carlota Castro Sánchez
### DATASET: 200-Financial-Indicators-of-US-stocks-2014-2018-
### URL: [kaggle] https://www.kaggle.com/cnic92/200-financial-indicators-of-us-stocks-20142018/code
## Resum
El datatset total està format per 5 datasets en format .csv:
2014_Financial_Data.csv
2015_Financial_Data.csv
2016_Financial_Data.csv
2017_Financial_Data.csv
2018_Financial_Data.csv

Cada conjunt de dades conté més de 200 indicadors financers (en total 225), que es troben habitualment en els documents de 10.000 que cada empresa cotitza en borsa publica anualment, per a una gran quantitat d'accions dels EUA (de mitjana, s'inclouen 4.000 accions a cada conjunt de dades).

Observacions importants sobre els conjunts de dades:

1. Falten alguns valors d'indicadors financers (cel·les nan), aquests van estar eliminats durant el data cleaning mitjançant dropna().

2. Hi ha valors atípics, és a dir, valors extrems que probablement són causats per errors d'escriptura. Diferents empreses es trobaben amb creixements de més de 500%. En aquests casos, com el que el seu creixement no resultava orgànic vaig optar per eliminar aquelles accions amb un creixement desproporcionat per a que no afectès al rendiment de les meves prediccions.

3. La tercera última columna, Sector, enumera el sector de cada acció. En efecte, a la borsa nord-americana cada empresa forma part d'un sector que la classifica en una macroàrea. Atès que s'han recollit tots els sectors (Materials bàsics, Serveis de comunicació, Consum cíclic, Consumidor defensiu, Energia, Serveis financers, Sanitat, Industrial, Immobiliari, Tecnologia i Utilitats).

4. La penúltima columna, PREU VAR [%], mostra el percentatge de variació del preu de cada acció durant l'any. Per exemple, si tenim en compte el conjunt de dades 2015_Financial_Data.csv, tindrem:

  - Més de 200 indicadors financers per a l'any 2015;
  - Variació del preu per cent per a l'any 2016 (és a dir, des del primer dia de negociació del gener de         2016 fins a l'últim dia de negociació del desembre de 2016).

5. L'última columna, classe, enumera una classificació binària per a cada estoc, on
  - Per a cada acció, si el valor de PREU VAR [%] és positiu, classe = 1. Des d'una perspectiva comercial, l'1 identifica aquelles accions que un comerciant hipotètic hauria de COMPRAR a principis d'any i vendre al final de l'any. Si pel contrari  el valor del PREU VAR [%] és negatiu, classe = 0. Des d'una perspectiva comercial, el 0 identifica aquelles accions que un comerciant hipotètic NO hauria de COMPRAR, ja que el seu valor disminuirà, és a dir, una pèrdua de capital.

Les dades no es torbaben normalitzades i va estar un procès a afegir, tot i que desprès de diferents probes no va semblar afectar el resultat de predicció. 

### Objectius del dataset
L'objectiu del dataset és aprendre a predir si una acció de la borse americana és una bona compra o no, mitjançant l'anlàisi del rendiment de l'empresa i amb uns indicadors estàndards. Per a arribar a aquestes conclusions, l'autor ens proporciona una sèrie d'atributs molt extensa com ara el sector al que pertany cada acció, la quantitat de deute de l'empresa, així com el percentatge de variació del preu de cada acció durant l'any. 
## Experiments
Durant aquesta pràctica hem realitzat diferents experiments.
### Preprocessat
Quines proves hem realitzat que tinguin a veure amb el pre-processat? com han afectat als resultats?
### Model
| Model | Hiperparametres | Mètrica | Temps |
| -- | -- | -- | -- |
| [Random Forest](link) | 100 Trees, XX | 57% | 100ms |
| Random Forest | 1000 Trees, XX | 58% | 1000ms |
| SVM | kernel: lineal C:10 | 58% | 200ms |
| -- | -- | -- | -- |
| [model de XXX](link al kaggle) | XXX | 58% | ?ms |
| [model de XXX](link al kaggle) | XXX | 62% | ?ms |
## Demo
Per tal de fer una prova, es pot fer servir amb la següent comanda
``` python3 demo/demo.py --input here ```
## Conclusions
El millor model que s'ha aconseguit ha estat...
En comparació amb l'estat de l'art i els altres treballs que hem analitzat....
## Idees per treballar en un futur
Crec que seria interesant indagar més en...
## Llicencia
El projecte s’ha desenvolupat sota llicència ZZZz.
