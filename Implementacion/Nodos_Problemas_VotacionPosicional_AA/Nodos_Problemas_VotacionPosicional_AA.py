
# coding: utf-8

# In[263]:

import pandas as pd
import numpy as np
import itertools as it
import networkx as nx
# Documentacion de la libreria: http://networkx.readthedocs.io/en/networkx-1.11/


# In[264]:

def filterData(df, isTraining, date):
    """
        Funcion que devuelve el conjunto de problemas que tienen status AC o PE
        Si isTraining es true, entonces la funcion sacara el training_set, si no, sacara el evaluation_set
        date es la fecha de particion
    """
    
    if isTraining:
        df = df[df['submissionDate'] < date]
        df = df.loc[df['status'].isin(['AC', 'PE'])]
    else:
        df = df[df['submissionDate'] >= date]
    
    

    return df


# In[265]:

def compareNodes(f_list, s_list):
    """
        Funcion que devuelve el numero de usuarios que han hecho ambos problemas
    """
    peso = len(np.intersect1d(f_list, s_list))
    
    return peso
    
def createLinks(prob_us_set, nodos):
    """
        Funcion que crea los enlaces del grafo a partir de la informacion contenida en el conjunto que se le
        pasa a la funcion
    """
    resultado = list() 
    
    # hago todas las posibles combinaciones de problemas
    for fst, snd in it.combinations(nodos, 2):
        # obtengo el peso pasando la lista de usuarios que ha hecho cada problema
        peso = compareNodes(prob_us_set[fst], prob_us_set[snd])
        if peso >= 1:
            resultado.append((fst, snd, peso))
            
            
            
    return resultado


# In[266]:

def filterWeight(weightUmbral, linksToFilter):
    """
        Funcion que filtra los enlaces de un grafo, para que el peso sea mayor o igual al dado
    """
    
    result = [(x, y, z) for (x, y, z) in linksToFilter if z >= weightUmbral]
    
    return result
    


# In[267]:

def create_graph_nx(list_nodes, list_links):
    """
        Funcion que crea un grafo de tipo Graph de la libreria NetworkX
        Construccion del grafo: http://networkx.readthedocs.io/en/networkx-1.11/tutorial/tutorial.html#what-to-use-as-nodes-and-edges
    """
    grafo = nx.Graph() # creo la variable grafo

    # incluyo los nodos del grafo 
    grafo.add_nodes_from(list_nodes)

    # se incluyen las tuplas de enlaces con el peso del enlace
    # es una lista de la forma [(Nodo1, Nodo2, peso), ......]
    grafo.add_weighted_edges_from(list_links)

    return grafo


# In[268]:

# MAIN
# ---------

# se guarda en la variable df (DataFrame) toda la base de datos
df = pd.read_csv('bbdd_orderbydate.csv')

# aqui quito los problemas que no existian despues de la fecha umbral
df = df[df['problem_id'] <= 511] 

# construyo el conjunto de entrenamiento
training_set = filterData(df, True, "2016-10-21 00:00:00")

print(len(training_set))

# obtengo los nodos del grafo:
nodes = training_set.problem_id.unique()


# creo un diccionario que va a tener a los problemas como keys y los valores seran los
# usuarios que han hecho ese problema
grouped = training_set.groupby('problem_id')['user_id'].apply(list)

print(len(nodes))
# print(grouped)


# In[269]:

print(training_set)


# In[270]:

# OBTENCION DEL EVALUATION_SET
# -------

# ahota saco el evaluation_set
evaluation_set = filterData(df, False, "2016-10-21 00:00:00")

print(evaluation_set)

# creo un diccionario que va a tener a los usuarios como keys y a los problemas que ha hecho como valores
# a partir del conjunto de entrenamiento
grouped_user_eval = evaluation_set.groupby('user_id')['problem_id'].apply(list)

# convierto la serie en un dataframe
df_users_eval = pd.DataFrame({'user_id':grouped_user_eval.index, 'list_problem_id':grouped_user_eval.values})

print(df_users_eval)


# In[271]:


# creo los enlaces a partir de la informacion de los nodos
links = createLinks(grouped, nodes)
# ahora filtro el grafo para que los enlaces solo tengan el peso que quiero
linksFiltered = filterWeight(5, links)

print(len(linksFiltered))

# aqui creo el grafo 
graph = create_graph_nx(nodes, linksFiltered)


# In[272]:

def apply_aa(row, graph):
    """
        Funcion que devuelve para cada par de nodos, el sumatorio de 1/log(N(z)), siendo N(z) el grado del nodo z para todo z 
        perteneciente al conjunto de nodos en comun de ese par de nodos
    """
    
    # obtengo un iterador de un solo elemento que tiene en la tercera posicion el valor de AA para el par de nodos
    value = nx.adamic_adar_index(graph, [(row['one'], row['two'])])
    
    value_aa = 0
    for u, v, p in value:
        # itero el iterador, guardando el valor de adar adamic
        value_aa = p
    
    return value_aa

def create_aa_data(graph, nodes):

    # Ahora voy a construir un DataFrame que tenga dos columnas con todas las posibles combinaciones de problemas, y otra 
    # columna con el valor de aa para ese par de problemas
    fst_column = list()
    snd_column = list()
    for fst, snd in it.combinations(nodes, 2):
        fst_column.append(fst)
        snd_column.append(snd)

    d = {'one' : fst_column,
        'two' : snd_column}
    dataFrame_aa = pd.DataFrame(d)


    # Aplico la funcion a cada fila
    dataFrame_aa['aa'] = dataFrame_aa.apply (lambda row: apply_aa(row, graph), axis=1)


    return dataFrame_aa


# In[273]:

aa_df = create_aa_data(graph, nodes)
print(aa_df)


# In[274]:

# creo un diccionario que va a tener a los usuarios como keys y a los problemas que ha hecho como valores
# a partir del conjunto de entrenamiento
grouped_user = training_set.groupby('user_id')['problem_id'].apply(list)

# convierto la serie en un dataframe
df_users = pd.DataFrame({'user_id':grouped_user.index, 'list_problem_id':grouped_user.values})

print(df_users)


# In[275]:

def lenProblemsDone(row, set_filter):
    """
        Funcion auxiliar que calcula cuanto problemas ha hecho cada usuario en un conjunto: training o evaluation
    """
    # saco el dataframe que contendra solo una fila con la lista de problemas que ha hecho el usuario
    df_filter = set_filter[set_filter['user_id'] == row['user_id']]
    
    if df_filter.empty:
        # si esta vacio, entonces es que el usuario no ha hecho problemas en ese conjunto
        return 0
    else:
        # sino, devuelvo la longitud de la lista de problemas
        return len(df_filter['list_problem_id'].iloc[0]) 
    
    


# In[276]:

# aqui voy a hacer el filtro de usuarios de forma que para hacer las recomendaciones solo tengamos en 
# cuenta aquellos usuarios que han hecho 5 o mas problemas tanto antes de la fecha limite como despues

# primero guardo la lista de usuarios
user_list = df.user_id.unique()

# la meto en un dataframe 
column_user_filter = {'user_id': user_list}
datraframe_user_filter = pd.DataFrame.from_dict(column_user_filter)


# ahora tengo que calcular para cada fila, el numero de problemas que han hecho en el training_set, evaluation_set
datraframe_user_filter['len_training'] = datraframe_user_filter.apply (lambda row: lenProblemsDone(row, df_users), axis=1)
datraframe_user_filter['len_evaluation'] = datraframe_user_filter.apply (lambda row: lenProblemsDone(row, df_users_eval), axis=1)
print(datraframe_user_filter)


# In[277]:

# ahora tengo que hacer el filtro en este dataframe, de forma que solo aparezcan las filas en las que len_training y 
# len_evaluation sea >=5
datraframe_user_filter = datraframe_user_filter[(datraframe_user_filter['len_training'] >= 5) & (datraframe_user_filter['len_evaluation'] >=5)]
print(datraframe_user_filter)

# aqui voy a guardar la lista de usuarios a los que voy a recomendar
user_list_to_recommend = sorted(datraframe_user_filter['user_id'].tolist())
print(user_list_to_recommend)
print(len(user_list_to_recommend))


# In[278]:

# ahora tengo que filtrar df_users para que solo contenga las filas en las que los usuarios
# pertenecen a la anterior lista

df_users = df_users[df_users['user_id'].isin(user_list_to_recommend)]
print(df_users)


# In[279]:

# creo el nuevo dataframe con los resultados 
column_problem_recomend = {'problem_id': nodes}
dataframe_problem_recomend = pd.DataFrame.from_dict(column_problem_recomend)

print(dataframe_problem_recomend)


# In[280]:

def getSimilarProblems(row, aa_df):
    """
        Funcion que devuelve la lista de los problemas similares del problema de la fila: aquellos que tienen un valor AA
        mayor a cero
    """
    # print(row['user_id'])
    
    # obtengo dos df con los usuarios que tienen usuarios vecinos con el usuario de la fila 
    column_result_one_tmp = aa_df[aa_df['one'] == row['problem_id']]
    column_result_one = column_result_one_tmp[column_result_one_tmp['aa'] > 0]
    column_result_two_tmp = aa_df[aa_df['two'] == row['problem_id']]
    column_result_two = column_result_two_tmp[column_result_two_tmp['aa'] > 0]
    
    # saco las listas de usuarios con usuarios comunes
    list_one = list(column_result_one['two'])
    list_two = list(column_result_two['one'])
    
    # la concateno sin tener en cuenta repeticiones, porque nunca va a haber
    list_aa = list_one + list_two
    
    # print(list_aa)
    
    if list_aa == []: # sino tiene vecinos en comun, pongo toda la lista de nodos
        list_aa = graph.nodes()
        list_aa.remove(row['problem_id']) # y elimino el nodo que estoy mirando
    
    # hago el filtro de los k mejores
    return list_aa


# In[281]:

dataframe_problem_recomend['neighbors'] = dataframe_problem_recomend.apply (lambda row: getSimilarProblems(row, aa_df), axis=1)

# aqui tengo la lista de usuarios con sus k usuarios similares
print(dataframe_problem_recomend)


# In[282]:

print(grouped)
# convierto la serie en un dataframe
df_users_recommend = pd.DataFrame({'problem_id':grouped.index, 'list_user_id':grouped.values})
print(df_users_recommend)


# In[283]:

def aa_value(one, two, graph):
    """
        Funcion que devuelve para ese par de nodos, el sumatorio de 1/log(N(z)), siendo N(z) el grado del nodo z para todo z 
        perteneciente al conjunto de nodos en comun de ese par de nodos
    """

    # obtengo un iterador de un solo elemento que tiene en la tercera posicion el valor de AA para el par de nodos
    value = nx.adamic_adar_index(graph, [(one, two)])
    # print(value)
    value_aa = -1
    for u, v, p in value:
        # print(p)
        # itero el iterador, guardando el valor de adar adamic
        value_aa = p

    return value_aa


# In[284]:

def getWeights(row, graph):
    """
        Funcion que calcula la suma de todos los valores de aa de todos los nodos con los que tiene aa
    """
    
    # primero obtengo la lista de los vecinos
    neighbors_list = row['neighbors']
    
    # print(neighbors_list)
    
    problem = row['problem_id']
    suma = 0
    
    for elem in neighbors_list:
        # print(suma)
        # print(graph[user][elem]['weight'])
        suma = suma + aa_value(problem, elem, graph)
        
    return suma


# In[285]:

def getPonderaciones(row, graph):
    """
        Funcion que calcula la ponderacion para cada problema vecino del de la fila
        Calculo la ponderacion diviendo el peso del enlace que enlaza cada problema con problem_id con la suma 
        total de los pesos
    """
    # primero obtengo la lista de los vecinos
    neighbors_list = row['neighbors']
    
    # obtengo la suma de pesos de los enlaces de esa lista
    total_weight = row['total_weight']
    
    # obtengo el id del usuario al que quiero recomendar
    problem = row['problem_id']
    
    lista_ponderaciones = list()
    
    for elem in neighbors_list:
        
        # obtengo el peso del enlace
        peso_enlace = aa_value(problem, elem, graph)
        
        if total_weight == 0:
            ponderacion = 0
        else:
            # hago la ponderacion
            ponderacion = peso_enlace/total_weight
        
        lista_ponderaciones.append(ponderacion)
        
    return lista_ponderaciones
    


# In[286]:

# ahora voy a incluir una nueva columna que tenga las ponderaciones (suma de todos los enlaces) de cada usuario
dataframe_problem_recomend['total_weight'] = dataframe_problem_recomend.apply (lambda row: getWeights(row, graph), axis=1)

print(dataframe_problem_recomend)


# In[287]:

# para cada usuario a "recomendar" para cada problema miro el numero de apariciones en su lista

dataframe_problem_recomend['score'] = dataframe_problem_recomend.apply (lambda row: getPonderaciones(row, graph), axis=1)

print(dataframe_problem_recomend)


# In[288]:


# elimino la columna de total_weight, ya que no interesa
del dataframe_problem_recomend['total_weight']

print(dataframe_problem_recomend)


# In[289]:

def getListUsersFromSimilarProblems(row, df_users_recommend):
    """
        Funcion que va a devolver por cada fila una lista procedente de listas la concatenacion de listas de usuarios que han
        hecho los problemas similares a ese. Además eliminara los usuarios que ya hayan hecho el problema
    """
    
    # obtengo la lista de problemas que ha hecho el usuario en cuestion
    list_problems_users = df_users_recommend[df_users_recommend['problem_id'] == row['problem_id']]
    list_problems_user = list(list_problems_users['list_user_id'])[0]
   
    # lista resultante 
    list_result = list(list())
    
    # obtengo la longitud de la lista de vecinos de ese problema
    list_neighbors = row['neighbors']
    k = len(list_neighbors)
    
    # recorro la lista de problemas vecinos 
    for i in range(0, k):
        # print(row['list_similar_users'][i])
        # aqui saco la lista de problemas que ha hecho el usuario similar
        list_users_df = df_users_recommend[df_users_recommend['problem_id'] == row['neighbors'][i]]
        lista_problemas_comprobar = list(list_users_df['list_user_id'])[0]
        
        # aqui hago el filtro para que no se incluyan los problemas que ya ha hecho el usuario
        list_problems = [x for x in lista_problemas_comprobar if x not in list_problems_user]
        
        # ahora incluyo la lista en la lista
        list_result.append(list_problems)
        # print(list_problems)
        # print(list_result)
        # print("---------------")
    
    return list_result


# In[290]:

dataframe_problem_recomend['lista_users_por_problema'] = dataframe_problem_recomend.apply(lambda row: getListUsersFromSimilarProblems(row, df_users_recommend), axis=1)

print(dataframe_problem_recomend)


# In[291]:

# ahora voy a separar cada problema_user para hacer la cuenta
# creo un nuevo dataframe que agrupa por el problema con su posible usuario
df_separation_neigh = dataframe_problem_recomend.groupby(['problem_id']).neighbors.apply(lambda x: pd.DataFrame(x.values[0])).reset_index().drop('level_1', axis = 1)

df_separation_neigh.columns = ['problem_id', 'neighbors', ]

print(df_separation_neigh)


# In[292]:

# creo un nuevo dataframe que agrupa por el usuario, poniendo en la columna de al lado, el score
df_separation_pond = dataframe_problem_recomend.groupby(['problem_id']).score.apply(lambda x: pd.DataFrame(x.values[0])).reset_index().drop('level_1', axis = 1)

df_separation_pond.columns = ['problem_id', 'score']

print(df_separation_pond)


# In[293]:

df_separation = df_separation_neigh

df_separation['score'] = df_separation_pond['score']

# creo ahora el nuevo dataframe gracias a los dos df anteriores que eran auxiliares
print(df_separation)


# In[294]:

# ahora voy a ordenar en funcion del score de mayor a menor para cada problema
# ahora lo que quiero es ordenar los vecinos por cada problema en funcion de su ponderacion
# primero ordeno por su valor de problem y luego por el de ponderacion, de forma que quedan ordenador por su valor de ponderacion
df_separation = df_separation.sort_values(by=['problem_id', 'score'], ascending=False)
print(df_separation)


# In[295]:

# elimino la columna ya que no interesa
del df_separation['score']

print(df_separation)


# In[296]:

# ahora tengo que hacer un nuevo dataframe con problem_id y vecino
# hago primero la agrupacion por problem_id
grouped_r = df_separation.groupby('problem_id')

# hago la agregacion en una lista 
df_aux = grouped_r.aggregate(lambda x:list(x))

print(df_aux)


# In[297]:

#print(sorted(nodes))

# voy a crear un nuevo dataframe con la columna problem_id y neighbors (con el anterior no se puede trabajar sin indices)
df_recommend = pd.DataFrame({'problem_id':sorted(nodes), 'neighbors':df_aux['neighbors'].tolist()})

print(df_recommend)


# In[298]:

def getUsersFromSimilarProblems(row, df_users_recommend):
    """
        Funcion que va a devolver por cada fila una lista procedente de la concatenacion de listas de usuarios que han
        realizado los problemas similares al de la fila. Además eliminara los usuarios que ya hayan realizado el problema.
    """
    
    # obtengo la lista de usuarios que han hecho el problema en cuestion
    list_problems_users = df_users_recommend[df_users_recommend['problem_id'] == row['problem_id']]
    list_problems_user = list(list_problems_users['list_user_id'])[0]
   
    # print(list_problems_user)

    # lista resultante de la concatenacion de las listas de usuarios de los problemas similares
    list_result = list()
    
    # obtengo la longitud de la lista de vecinos de ese usuario
    list_neighbors = row['neighbors']
    k = len(list_neighbors)
    
    # recorro la lista de usuarios vecinos 
    for i in range(0, k):
        # print(row['list_similar_users'][i])
        # aqui saco la lista de usuarios que han hecho el problema similar
        list_problems_df = df_users_recommend[df_users_recommend['problem_id'] == row['neighbors'][i]]
        lista_problemas_comprobar = list(list_problems_df['list_user_id'])[0]
        
        # print("----------")
        #print(lista_problemas_comprobar)
        
        # aqui hago el filtro para que no se incluyan los usuarios que ya ha hechon el problema
        list_users = [x for x in lista_problemas_comprobar if x not in list_problems_user]
        
        # ahora concateno el resultado
        list_result = list_result + list_users
        # print(list_problems)
        # print(list_result)
        # print("---------------")
    
    return list_result


# In[299]:

# ahora para cada problema, hacer una lista de los usuarios similares a los que han realizado ese problema, 
# que no sean usuarios que han realizado ya ese problema
df_recommend['list_users'] = dataframe_problem_recomend.apply (lambda row: getUsersFromSimilarProblems(row, df_users_recommend), axis=1)
print(df_recommend)


# In[300]:

def delRepetitions(row):
    """
        Funcion auxiliar para evitar que salgan repeticiones en las recomendaciones. Saco la lista de posibles 
        recomendaciones con valores unicos
    """
    conjunto_vacio = set()
    
    # esto sirve para que se haga mas rapido la comprobacion de si el elemento esta en la lista o no
    function_add = conjunto_vacio.add
    
    # hago la lista intensional, para mantener el orden dado en la lista original
    return [x for x in row['list_users'] if not (x in conjunto_vacio or function_add(x))]


# In[301]:

# tengo que contar el numero de apariciones de cada usuario en la lista de usuarios

# voy a sacar primero una lista sin repeticiones
# ahora voy a crear una nueva columna que contenga la lista de usuarios sin repeticiones
df_recommend['lista_users_unique'] = df_recommend.apply(lambda row: delRepetitions(row), axis=1)

print(df_recommend)


# In[302]:

# ahora voy a separar cada user-problema_a_recomendar para hacer la cuenta
# creo un nuevo dataframe que agrupa por el primer problema y tiene su posible recomendacion
new_df_separation = df_recommend.groupby(['problem_id']).lista_users_unique.apply(lambda x: pd.DataFrame(x.values[0])).reset_index().drop('level_1', axis = 1)

new_df_separation.columns = ['problem_id', 'possible_user']

print(new_df_separation)


# In[303]:

def getScoring(row, dataframe_problem_recomend):
    """
        Funcion en la que por cada usuario-problema, hace el scoring para ese problema (para ese usuario)
        sumando todas las ponderaciones de ese problema para ese usuario
    """
     # obtengo la lista de listas de usuarios que han hecho los vecinos del problema
    list_problems_users_df = dataframe_problem_recomend[dataframe_problem_recomend['problem_id'] == row['problem_id']]
    list_user_per_problem = list(list_problems_users_df['lista_users_por_problema'])[0]
    
    # saco tambien la lista de scores por vecinos
    list_scores_per_problem = list(list_problems_users_df['score'])[0]
    
    # print(list_scores_per_user)
    
    # obtengo el usuario del que quiero calcular el scoring
    user = row['possible_user']
    # print(problem)
    
    suma = 0
    k = len(list_user_per_problem)
    
    # por cada lista de la lista, saco la ponderacion de su correspondiente usuario (el usuario que lo ha hecho)
    # si problem aparece en la lista, sumo esa ponderacion
    for i in range(0, k):
        lista_a_comprobar = list_user_per_problem[i]
        # print(lista_a_comprobar)
        if user in lista_a_comprobar:
            # print(list_scores_per_user[i])
            suma = suma + (1/(i+1))
    
    return suma


# In[304]:

# ahora voy a calcular el scoring para cada problema-usuario
new_df_separation['score'] = new_df_separation.apply(lambda row: getScoring(row, dataframe_problem_recomend), axis=1)

print(new_df_separation)


# In[305]:

# ahora voy a ordenar en funcion del score de mayor a menor para cada usuario
# ahora lo que quiero es ordenar los problemas por cada usuario en funcion de su ponderacion
# primero ordeno por su valor de user y luego por el de ponderacion, de forma que quedan ordenador por su valor de ponderacion
new_df_separation = new_df_separation.sort_values(by=['possible_user', 'score'], ascending=False)
print(new_df_separation)


# In[306]:

# ahora tengo que hacer un nuevo dataframe con usuario y problema, y una lista de recommendation 

# hago primero la agrupacion por usuario
grouped_r = new_df_separation.groupby('possible_user')

# hago la agregacion en una lista 
df_recommend = grouped_r.aggregate(lambda x:list(x))

#print(df_recommend)

# vuelvo a crear la estructura de los datos para poder trabajar con ellos
df_recommend = pd.DataFrame({'user_id':df_recommend.index.values, 'problem_id':df_recommend['problem_id'].tolist()})
print(df_recommend)


# In[307]:

def getProblems(row, df_recommend):
    """
        Funcion que copia los posibles problemas a recomendar para los usuarios a los que tengo que recomendar
    """
    
    # saco la lista de problemas 
    df_lista_problemas_original = df_recommend[df_recommend['user_id'] == row['user_id']]
    lista_problemas_original = list(df_lista_problemas_original['problem_id'])[0]
    
    return lista_problemas_original
    


# In[308]:

# df_users # en este dataframe estan los usuarios a los que quiero recomendar

# ahora voy a copiar los problemas para los usuarios a los que tengo que recomendar
df_users['recommendation'] = df_users.apply (lambda row: getProblems(row, df_recommend), axis=1)

print(df_users)


# In[309]:

# elimino la columna de los problemas, ya que no interesa
del df_users['list_problem_id']

print(df_users)


# In[310]:

def getKrecomFinal(row, k):
    """
        Funcion que saca las k mejores recomendaciones para el usuario
        Lo que hace es coger los primeros k valores de la lista de recomendaciones
    """
    return row['recommendation'][:k]


# In[311]:

k = 10

#print(list(range(1, k+1)))

rango = list(range(1, k+1))

for i in rango:
    name_column = 'k_recommendation_' + str(i)
    #print(name_column)
    # ahora saco los k mejores problemas para cada usuario
    df_users[name_column] = df_users.apply(lambda row: getKrecomFinal(row, i), axis=1)

# he incluido una columna para cada cada k recomendacion 
print(df_users)


# In[312]:

# elimino las columnas que no me interesan
del df_users['recommendation']

print(df_users)


# In[313]:

# ahora tengo que filtrar df_users_eval para que solo contenga las filas de los usuarios a los que hay que recomendar

df_users_eval_filter = df_users_eval[df_users_eval['user_id'].isin(user_list_to_recommend)]
print(df_users_eval_filter)


# In[314]:

list_eval_problems = df_users_eval_filter['list_problem_id'].tolist()


# meto toda la informacion en un dataframe para obtener las metricas
set_df_metric = {'user_id': user_list_to_recommend, 'eval_problems': list_eval_problems}

for i in rango:
    name_column = 'k_recommendation_' + str(i)
    list_recom_problems = list()
    list_recom_problems = df_users[name_column].tolist()
    
    name_column_metric = 'recom_problems_' + str(i)
    set_df_metric[name_column_metric] = list_recom_problems

# he generado un nuevo dataframe con todas las recomendaciones (desde 1 a k) y los problemas realizados en 
# el evaluation set para cada usuario al que quiero recomendar

metric_df = pd.DataFrame.from_dict(set_df_metric)

print(metric_df)


# In[315]:

def one_hit(row, i):
    """
        Funcion que implementa la metrica one hit. Devuelve un 1 si para un usuarios dado, al menos uno
        de los problemas que se le ha recomendado ha sido realizado por ese usuario en el evaluation_set. 
        Cero si no hay ningun problema de los recomendados que haya sido realizado por el usuario
    """
    name_column = 'recom_problems_' + str(i)
    num_problems_common = np.intersect1d(row[name_column], row['eval_problems'])
    
    if len(num_problems_common) >= 1:
        return 1
    else:
        return 0


# In[316]:

def mrr(row, i): 
    """
        Funcion que va a implementar la metrica de evaluacion mrr:
        mrr = 1/ranki, donde ranki es la posicion del primer item correcto
    """
    name_column = 'recom_problems_' + str(i)
    num_problems_common = np.intersect1d(row[name_column], row['eval_problems'])
    
    if len(num_problems_common) >= 1:

        # hago la busqueda del primer elemento que esta en la lista de recomendados
        fst_correct_item = -1
        encontrado = False
        i = 0
        while (i < len(row[name_column])) and (encontrado == False):
            if row[name_column][i] in row['eval_problems']:
                # fst_correct_item = row['recom_problems'][i]
                # print(fst_correct_item)
                ranki = i + 1
                encontrado = True
            else:
                i = i + 1
                
        return (1/ranki)

    else:
        return 0


# In[317]:

def precision(row, i):
    """
        Funcion que va a implementar la metrica precision en k: 
        (cuantos de los realizados por el usuario estan entre los recomendados) / todos los recomendados
    """
    name_column = 'recom_problems_' + str(i)
    num_problems_common = np.intersect1d(row[name_column], row['eval_problems'])
    
    # print(num_problems_common)
    
    return (len(num_problems_common)/len(row[name_column]))


# In[318]:

def recall(row, i):
    """
        Funcion que implementa la metrica recall
        (cuantos de los realizados por el usuario estan entre los recomendados) / todos los evaluados
    """
    name_column = 'recom_problems_' + str(i)
    num_problems_common = np.intersect1d(row[name_column], row['eval_problems'])
    
    # print(num_problems_common)
    
    return (len(num_problems_common)/len(row['eval_problems']))


# In[319]:

def f1(row, i):
    """
        Funcion que calcula el f1 en funcion de precision y recall
    """
    name_column_prec = 'precision_' + str(i)
    name_column_rec = 'recall_' + str(i)
    denominador = row[name_column_prec] + row[name_column_rec]
    
    if denominador == 0:
        return 0
    else:
        return (2 * row[name_column_prec] * row[name_column_rec]) / denominador


# In[320]:

for i in rango:
    name_one_hit = 'one_hit_' + str(i)
    name_mrr = 'mrr_' + str(i)
    name_precision = 'precision_' + str(i)
    name_recall = 'recall_' + str(i)
    name_f1 = 'f1_' + str(i)
    
    # ahora voy a calcular una metrica para cada usuario
    metric_df[name_one_hit] = metric_df.apply(lambda row: one_hit(row, i), axis=1)
    metric_df[name_mrr] = metric_df.apply(lambda row: mrr(row, i), axis=1)
    metric_df[name_precision] = metric_df.apply(lambda row: precision(row, i), axis=1)
    metric_df[name_recall] = metric_df.apply(lambda row: recall(row, i), axis=1)
    metric_df[name_f1] = metric_df.apply(lambda row: f1(row, i), axis=1)

print(metric_df)


# In[321]:



f = open("C:/hlocal/TFM/vot_simple", 'a')

# calculo la media de las metricas
for i in rango:
    name_one_hit = 'one_hit_' + str(i)
    name_mrr = 'mrr_' + str(i)
    name_precision = 'precision_' + str(i)
    name_recall = 'recall_' + str(i)
    name_f1 = 'f1_' + str(i)
    
    result_one_hit = metric_df[name_one_hit].mean()
    result_precision = metric_df[name_precision].mean()
    result_mrr = metric_df[name_mrr].mean()
    result_recall = metric_df[name_recall].mean()
    result_f1 = metric_df[name_f1].mean()

    # lo muestro por consola
    
    print(i)
    print("###########")
    print("One hit ----------")
    print(result_one_hit)
    print("Precision ----------")
    print(result_precision)
    print("Mrr  ----------")
    print(result_mrr)
    print("Recall  ----------")
    print(result_recall)
    print("F1  ----------")
    print(result_f1)
    
    
    f.write(str(result_one_hit) + '\t' + str(result_precision) + '\t' + str(result_mrr) + '\t' + str(result_recall) + '\t' +  str(result_f1) + '\n') 

    
f.close()    
    


# In[ ]:


