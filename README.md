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
El primer que he realiztat ha estat la neteja de les dades  de cadascun dels datasets per separat donat que es tracta d'un dataset molt extens. Per a portar aquesta acció a terme he realitzat diferent mètodes:
1. El primer que fem es mrar PREU VAR % per tal de veure quines accions han tingut un creixement inorgànic de més de 500% i els eliminem.
2. Eliminem aquells valors que siguin buits(nans més del 50%) o tinguing una gran quantitat de zeros (més del 60% en el nostre cas)
3. Ens encarreguem també de les dades amb valors extrems (outliers).
4. Per últim emplenem aquells valors nans que hagin quedat amb la mitjana de la seva columna, però només tenint en compte els valors que pertanyin al mateix sector.

El següent pas ha estat ajuntar tots els dataset en un (data) per a poder predir a partir de les dades de 4 anys anteriors els valors de les accions de 2018.

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
El millor model que s'ha aconseguit ha estat Gradient Boosting amb un accuracy a les prediccions de 65% que supera el 50% que es demana per a considerar les prediccions d'un model, millor que si es fa de forma aleatòria.

Les mesures de rendibilitat no donen gaire informació sobre les empreses amb les majors variacions positives de preu l'any següent. En general, és important centrar-se en empreses amb EPS positius i passius baixos/actius totals elevats (balanços saludables), així com la quantitat total d'actius qeue tinguin. quests factors només minimitzen el risc de perdre diners, el rendiment passat encara no és una mètrica del tot fiable per al futur
Es veu clarament que hi ha uns sectors on el rendiment és molt major que en altres (fàcil de veure a l'EDA) i això pot canviar anualment amb els cicles econòmics. Les 5 variables financeres que el model d'aprenentatge automàtic considera importants: EPS, Actius totals, Marge de benefici net i Rendiment del capital.

## Idees per treballar en un futur
Crec que seria interesant indagar més en...
## Llicencia
El projecte s’ha desenvolupat sota llicència ZZZz.
