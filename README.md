## Annotations
1. when creating the graph must be change the way to assign attributes, for now I'm only adding attributes to nodes and not to the graph thus methods like *nx.get_node_attribute() DO NOT WORK!*
2. initialization for first fixed values is done inside agents thus in the DHGrid the worflow is standardized it only get and set values and steps the simulators every time
3. per ora non ho modellato gli storages come agenti: nel momento in cui lo farò dovrò considerare che nei gather di portate e temperature dovrò invocare anche loro negli step appropriati
4. **NB** i nodi sono modellati con (T_in ,T_out) se nodi temrinali e con (T_mandata, T_ritorno) se nodi centrali (e.g substations)
5. **DOMANDA** : ma se partiamo dalla conoscenza delle portate in utenza e non da quelle in centrale per calcolarci le portate in centrale nel caso ci fossero più centrali come facciamo?? dobbiamo configurare le porzioni percentauali del totale per spartirlo? o c'è un modo più intelligente 
6. **NB** constraints:

   - le reti di distribuzione hano una e una sola stazione BCT (una rete di distribuzione è la porzione a valle di una BCT)
   - per ora funziona con solo una centrale 
#GRAPH STRUCTURE
The whole grid, composed of transport and distribution, is represented using a graph structure:
   
      node_attr : { 
                     'type': str ('Gen'/ 'free'/ 'inner'/ 'BCT'/ 'Utenza'),
                     'group': str ('dist_n'/ 'transp'/'T-D'), #where T-D are the names of the joined groups),
                     'storages: list[] (with unique name of the storages),
                     'connection': list[] (with unique name of connected node)
                  }
      
      edge_attr : {
                     'lenght': float (lenght of the pipe [m]),
                     'D': float (diameter of the pipe [m]),
                     'group': str ('dist_n'/ 'transp' )
                     'NB': int (position index of the specific branch in the sample data matrix),
                  }

**nodes naming**: groupname_index (e.g. dist_0_124, transp_23)
                  


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
   

