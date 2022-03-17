##possible upgrades
1. check automatico che le portate di accumulo in discarica non completino al 100% la domanda in modo che la portate di generazione non vada in negativo oppure trovare un modo per cambiare i versi
2. trovare il modo per gestire le portate di storage dinamicamente ossia : se scaricando secondo schedule ho dei momenti in cui copro più della domanda non tiro fuori tutto ma tiro fuori di meno e allunco il tempo di schedule
##problems
1. storages not working properly maybe Incidence matrices must change when active
2. G (vettore portate nelle branches per gli storgae quando sono in carica scarico viene sbagliato perchè matrice di incidenza è al contrario, va modificata)
## Annotations
1. when creating the graph must be change the way to assign attributes, for now I'm only adding attributes to nodes and not to the graph thus methods like *nx.get_node_attribute() DO NOT WORK!*
2. initialization for first fixed values is done inside agents thus in the DHGrid the worflow is standardized it only get and set values and steps the simulators every time
3. per ora non ho modellato gli storages come agenti: nel momento in cui lo farò dovrò considerare che nei gather di portate e temperature dovrò invocare anche loro negli step appropriati
4. **NB** i nodi sono modellati con (T_in ,T_out) se nodi temrinali e con (T_mandata, T_ritorno) se nodi centrali (e.g substations)
5. **DOMANDA** : ma se partiamo dalla conoscenza delle portate in utenza e non da quelle in centrale per calcolarci le portate in centrale nel caso ci fossero più centrali come facciamo?? dobbiamo configurare le porzioni percentauali del totale per spartirlo? o c'è un modo più intelligente 
6. **NB** constraints:
   - le reti di distribuzione hano una e una sola stazione BCT (una rete di distribuzione è la porzione a valle di una BCT)
   - per ora funziona con solo una centrale 

## Problems
1. **agent creation**: troppo lenta... 300s circa

#SCENARIO  DATA STRUCTURE
      scenario: {
               'complete_graph': DiGraph, #(grafo di tutta la rete)
               'group_list': list, #(e.g. ['transp', 'dist_0'...]) list of all the sub group names
               'dist_0':
                        {
                           'graph': DiGraph
                           'Ix': np.matrix
                           'NN': int
                           'NB': int
                           'L': array
                           'D': array
                           'D_ext': array
                           'nodi_immissione': list
                           'nodi_estrazione': list
                           'edge_list': list
                           'node_list': list
                        }

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
   

