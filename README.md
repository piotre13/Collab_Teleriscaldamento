## Annotations


#GRAPH STRUCTURE

#CALCULATION WORKFLOW
## MANDATA
1. recupero delle portate ai nodi estremi (leafs, utenze, accumuli) portate in ingresso per questi nodi

2. create G_ext vettore [NN] con portate dei nodi estremi e portate negative ai nodi estremi di inlet (e.g substations)

3. equazione di continuità per stabilire portate in tutti i nodi
   * input: G_ext (vettore con portate solo ai nodi estremi inlet outlet)
   * output: G (vettore con tutte le portate per nodo)

4. recupero delle temperature di inlet (e.g sottostazioni, accumuli)

5. calcolo delle matrici [ M, K, f ]
   * input: G, G_ext an inlet temperatures (e.g. substations, storages)
   * output: [ M, K, f ]

6. calculate temperatures for each node
   * input: [ M, K, f ], Temperatures at t-1
   * output: Temperatures at t
## RITORNO

1. recupero delle portate di inlet (e.g. utenze)
   * NB in questo caso sono le stesse di prima ma invertite di segno

2. recupero delle temperature di inlet (e.g. utenze)
   * NB nel caso della distribuzione le T delle utenze sono calcolate tramite il calcolo della potenza assorbita

3. calcolo delle matrici [ M, K, f ]
   * input: G, G_ext an inlet temperatures (e.g. substations, storages)
   * output: [ M, K, f ]

4. calculate temperatures for each node
   * input: [ M, K, f ], Temperatures at t-1
   * output: Temperatures at t   
   

