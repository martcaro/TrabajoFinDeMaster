
# coding: utf-8

# In[394]:

import pandas as pd
import numpy as np
import itertools as it
from datetime import datetime
import networkx as nx
# Documentacion de la libreria: http://networkx.readthedocs.io/en/networkx-1.11/

from operator import itemgetter
from itertools import groupby


# In[395]:

def filterProblems(df, isTraining, date):
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


# In[396]:

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


# In[397]:

def filterWeight(weightUmbral, linksToFilter):
    """
        Funcion que filtra los enlaces de un grafo, para que el peso sea mayor o igual al dado
    """
    
    result = [(x, y, z) for (x, y, z) in linksToFilter if z >= weightUmbral]
    
    return result
    


# In[398]:

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


# In[399]:

# MAIN
# ---------

# se guarda en la variable df (DataFrame) toda la base de datos
df = pd.read_csv('bbdd_orderbydate.csv')

# aqui quito los problemas que no existian despues de la fecha umbral
df = df[df['problem_id'] <= 511] 

# construyo el conjunto de entrenamiento
training_set = filterProblems(df, True, "2016-10-21 00:00:00")

print(len(training_set))

# obtengo los nodos del grafo:
nodes = training_set.problem_id.unique()


# creo un diccionario que va a tener a los problemas como keys y los valores seran los
# usuarios que han hecho ese problema
grouped = training_set.groupby('problem_id')['user_id'].apply(list)




# In[400]:

print(training_set)


# In[401]:

# OBTENCION DEL EVALUATION_SET
# -------

# ahota saco el evaluation_set
evaluation_set = filterProblems(df, False, "2016-10-21 00:00:00")

print(evaluation_set)

# creo un diccionario que va a tener a los usuarios como keys y a los problemas que ha hecho como valores
# a partir del conjunto de entrenamiento
grouped_user_eval = evaluation_set.groupby('user_id')['problem_id'].apply(list)

# convierto la serie en un dataframe
df_users_eval = pd.DataFrame({'user_id':grouped_user_eval.index, 'list_problem_id':grouped_user_eval.values})

print(df_users_eval)


# In[402]:


# creo los enlaces a partir de la informacion de los nodos
links = createLinks(grouped, nodes)
# ahora filtro el grafo para que los enlaces solo tengan el peso que quiero
linksFiltered = filterWeight(5, links)

print(len(linksFiltered))

# aqui creo el grafo 
graph = create_graph_nx(nodes, linksFiltered)


# In[403]:

def apply_wcn(row, graph):
    """
        Funcion que devuelve el numero de vecinos en comun de esos dos nodos
    """
    cn_list = nx.common_neighbors(graph, row['one'], row['two'])
    
    value_wcn = sum([graph[row['one']][x]['weight'] + graph[row['two']][x]['weight'] for x in cn_list])
    
    return value_wcn

def create_wcn_data(graph, nodes):

    # Ahora voy a construir un DataFrame que tenga dos columnas con todas las posibles combinaciones de problemas, y otra 
    # columna con el valor de wcn para ese par de problemas
    fst_column = list()
    snd_column = list()
    for fst, snd in it.combinations(nodes, 2):
        fst_column.append(fst)
        snd_column.append(snd)

    d = {'one' : fst_column,
        'two' : snd_column}
    dataFrame_wcn = pd.DataFrame(d)


    # Aplico la funcion a cada fila
    dataFrame_wcn['wcn'] = dataFrame_wcn.apply (lambda row: apply_wcn(row, graph), axis=1)


    return dataFrame_wcn


# In[404]:

# Hasta aqui se hace la creacion del grafo
wcn_df = create_wcn_data(graph, nodes)
print(wcn_df)


# In[405]:

# creo un diccionario que va a tener a los usuarios como keys y a los problemas que ha hecho como valores
# a partir del conjunto de entrenamiento
grouped_user = training_set.groupby('user_id')['problem_id'].apply(list)

# convierto la serie en un dataframe
df_users = pd.DataFrame({'user_id':grouped_user.index, 'list_problem_id':grouped_user.values})

print(df_users)


# In[406]:

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
    
    


# In[407]:

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


# In[408]:

# ahora tengo que hacer el filtro en este dataframe, de forma que solo aparezcan las filas en las que len_training y 
# len_evaluation sea >=5
datraframe_user_filter = datraframe_user_filter[(datraframe_user_filter['len_training'] >= 5) & (datraframe_user_filter['len_evaluation'] >=5)]
print(datraframe_user_filter)

# aqui voy a guardar la lista de usuarios a los que voy a recomendar
user_list_to_recommend = sorted(datraframe_user_filter['user_id'].tolist())
print(user_list_to_recommend)
print(len(user_list_to_recommend))


# In[409]:

# ahora tengo que filtrar df_users para que solo contenga las filas en las que los usuarios
# pertenecen a la anterior lista

df_users = df_users[df_users['user_id'].isin(user_list_to_recommend)]
print(df_users)


# In[410]:

# con esto creo un dataframe que separa el dataframe anterior
# lista de usuario - problema en el conjunto de entrenamiento
df_new = df_users.groupby('user_id').list_problem_id.apply(lambda x: pd.DataFrame(x.values[0])).reset_index().drop('level_1', axis = 1)

df_new.columns = ['user_id','problem']

print(df_new)


# In[411]:

def getKrecommendations(row, wcn_df, df_users, k):
    """
        Funcion que devuelve la lista de k mejores problemas para el usuario dado teniendo en cuenta que 
        las recomendaciones no son problemas que ya haya realziado el usuario
    """
    #column_result = wcn_df[wcn_df['one'] == 10].sort_values('wcn', ascending=False)
    # user_list = df_users[df_users['user_id'] == 6]
    
   
    
    # primero saco los dos dataframes con problemas que se pueden recomendar
    column_result_one = wcn_df[wcn_df['one'] == row['problem']]
    column_result_two = wcn_df[wcn_df['two'] == row['problem']]
    
    
    tmp1 = column_result_two['two'].tolist()
    tmp2 = column_result_two['one'].tolist()
    tmp3 = column_result_two['wcn'].tolist()
   
    # creo un nuevo df
    df_tmp = pd.DataFrame({'one':tmp1, 'two':tmp2, 'wcn': tmp3})
    
    frames = [column_result_one, df_tmp]
    
    # concateno los resultados
    column_result_tmp = pd.concat(frames)
    
    # ordeno los problemas que se pueden recomendar
    column_result_tmp2 = column_result_tmp.sort_values('wcn', ascending=False)
     
    tmp1 = column_result_tmp2['one'].tolist()
    tmp2 = column_result_tmp2['two'].tolist()
    tmp3 = column_result_tmp2['wcn'].tolist()
    
    # creo un nuevo df     
    column_result = pd.DataFrame({'one':tmp1, 'two':tmp2, 'wcn': tmp3})
    
    # con esto consigo sacar la lista de problemas que ha realizado ese usuario
    user_list = df_users[df_users['user_id'] == row['user_id']]
    user_list = user_list['list_problem_id'].iloc[0] 
    
    # ahora filtro la columna para que los problemas recomendados no los haya hecho ya el usuario
    column_result = column_result[column_result['two'].isin(user_list) == False]
    
    return (column_result['two'].head(k)).tolist()
    


# In[412]:

# Aplico la funcion a cada fila 
k = 1
df_new['list_recommendations'] = df_new.apply (lambda row: getKrecommendations(row, wcn_df,  df_users, k), axis=1)
print(df_new)


# In[413]:

# creo un nuevo dataframe que agrupa por el primer problema 
df_separation = df_new.groupby(['user_id', 'problem']).list_recommendations.apply(lambda x: pd.DataFrame(x.values[0])).reset_index().drop('level_2', axis = 1)

df_separation.columns = ['user_id','problem', 'recommendation']

print(df_separation)


# In[414]:

def asignawcnvalue(row, wcn_df):
    """
        Funcion que devuelve una columna con los valores de wcn asociados a cada fila
    """
    
    # print(wcn_df.one == row['problem'])
    # print(wcn_df.two)
    # print(row['recommendation'])
    # print(wcn_df.two == row['recommendation'])
    # print((wcn_df.one == row['problem']) & (wcn_df.two == row['recommendation']))
    
    new_df = wcn_df[(wcn_df.one == row['problem']) & (wcn_df.two == row['recommendation'])]
    
    if new_df.empty:
        new_df = wcn_df[(wcn_df.two == row['problem']) & (wcn_df.one == row['recommendation'])]
    
    return new_df.iloc[0]['wcn']


# In[415]:

df_separation['wcn'] = df_separation.apply(lambda row: asignawcnvalue(row, wcn_df), axis=1)
print(df_separation)


# In[416]:

# ahora lo que quiero es ordenar los problemas por cada usuario en funcion de su wcn
# primero ordeno por su valor de user y luego por el de wcn, de forma que quedan ordenador por su valor wcn
df_separation = df_separation.sort_values(by=['user_id', 'wcn'], ascending=False)
print(df_separation)


# In[417]:

# ahora tengo que hacer un nuevo dataframe con usuario, problema, y una lista de recommendation 
# (los tres primeros ya que estan ordenados por wcn)

# hago primero la agrupacion por usuario
grouped_r = df_separation.groupby('user_id')

# hago la agregacion en una lista 
df_recommend = grouped_r.aggregate(lambda x:list(x))

print(df_recommend)


# In[418]:

# para sacar el dataframe final con user - krecomendaciones
del df_recommend['problem']
del df_recommend['wcn']


# In[419]:

def delRepetitions(row):
    """
        Funcion auxiliar para evitar que salgan repeticiones en las recomendaciones. Saco la lista de posibles 
        recomendaciones con valores unicos
    """
    conjunto_vacio = set()
    
    # esto sirve para que se haga mas rapido la comprobacion de si el elemento esta en la lista o no
    function_add = conjunto_vacio.add
    
    # hago la lista intensional, para mantener el orden dado en la lista original
    return [x for x in row['recommendation'] if not (x in conjunto_vacio or function_add(x))]


# In[420]:

# ahora voy a aplicar una funcion a la lista de posibles recomendaciones, para quitar los repetidos
df_recommend['recommendation'] = df_recommend.apply(lambda row: delRepetitions(row), axis=1)
print(df_recommend)


# In[421]:

def getKrecomFinal(row, k):
    """
        Funcion que saca las k mejores recomendaciones para el usuario
        Lo que hace es coger los primeros k valores de la lista de recomendaciones
    """
    if k == 1:
        value = list()
        value.append(row['recommendation'][:k])
        return value
    else:
        return row['recommendation'][:k]


# In[422]:

# ahora saco los k mejores problemas para cada usuario
df_recommend['recommendation'] = df_recommend.apply(lambda row: getKrecomFinal(row, k), axis=1)

print(df_recommend)


# In[423]:

# hago el filtro para los usuarios a los que tengo que recomendar
df_users_eval = df_users_eval[df_users_eval['user_id'].isin(user_list_to_recommend)]
print(df_users_eval)


# In[424]:

# primero voy a ordenar la lista de usuarios a recomendar
user_list_to_recommend.sort()

list_eval_problems = df_users_eval['list_problem_id'].tolist()
list_recom_problems = df_recommend['recommendation'].tolist()


# meto toda la informacion en un dataframe para obtener las metricas
set_df_metric = {'user_id': user_list_to_recommend, 'eval_problems': list_eval_problems, 'recom_problems': list_recom_problems}
metric_df = pd.DataFrame.from_dict(set_df_metric)

print(metric_df)


# In[425]:

def one_hit(row):
    """
        Funcion que implementa la metrica one hit. Devuelve un 1 si para un usuarios dado, al menos uno
        de los problemas que se le ha recomendado ha sido realizado por ese usuario en el evaluation_set. 
        Cero si no hay ningun problema de los recomendados que haya sido realizado por el usuario
    """
    num_problems_common = np.intersect1d(row['recom_problems'], row['eval_problems'])
    
    if len(num_problems_common) >= 1:
        return 1
    else:
        return 0


# In[426]:

def mrr(row): 
    """
        Funcion que va a implementar la metrica de evaluacion mrr:
        mrr = 1/ranki, donde ranki es la posicion del primer item correcto
    """

    num_problems_common = np.intersect1d(row['recom_problems'], row['eval_problems'])
    
    if len(num_problems_common) >= 1:

        # hago la busqueda del primer elemento que esta en la lista de recomendados
        fst_correct_item = -1
        encontrado = False
        i = 0
        while (i < len(row['recom_problems'])) and (encontrado == False):
            if row['recom_problems'][i] in row['eval_problems']:
                # fst_correct_item = row['recom_problems'][i]
                # print(fst_correct_item)
                ranki = i + 1
                encontrado = True
            else:
                i = i + 1
                
        return (1/ranki)

    else:
        return 0


# In[427]:

def precision(row):
    """
        Funcion que va a implementar la metrica precision en k: 
        (cuantos de los realizados por el usuario estan entre los recomendados) / todos los recomendados
    """
    
    num_problems_common = np.intersect1d(row['recom_problems'], row['eval_problems'])
    
    # print(num_problems_common)
    
    return (len(num_problems_common)/len(row['recom_problems']))


# In[428]:

def recall(row):
    """
        Funcion que implementa la metrica recall
        (cuantos de los realizados por el usuario estan entre los recomendados) / todos los evaluados
    """
    num_problems_common = np.intersect1d(row['recom_problems'], row['eval_problems'])
    
    # print(num_problems_common)
    
    return (len(num_problems_common)/len(row['eval_problems']))


# In[429]:

def f1(row):
    """
        Funcion que calcula el f1 en funcion de precision y recall
    """
    denominador = row['precision'] + row['recall']
    
    if denominador == 0:
        return 0
    else:
        return (2 * row['precision'] * row['recall']) / denominador


# In[430]:

# ahora voy a calcular una metrica para cada usuario
metric_df['one_hit'] = metric_df.apply(lambda row: one_hit(row), axis=1)
metric_df['mrr'] = metric_df.apply(lambda row: mrr(row), axis=1)
metric_df['precision'] = metric_df.apply(lambda row: precision(row), axis=1)
metric_df['recall'] = metric_df.apply(lambda row: recall(row), axis=1)
metric_df['f1'] = metric_df.apply(lambda row: f1(row), axis=1)
print(metric_df)


# In[431]:

# Para crear un archivo grafo para GEPHI


# creo los enlaces a partir de la informacion de los nodos
# links = createLinks(grouped, nodes)


# aqui creo el grafo 
# graph = create_graph_nx(nodes, links)
# nx.write_gexf(graph,"grafo.gexf")


# In[432]:

# calculo la media de las metricas

result_one_hit = metric_df['one_hit'].mean()
result_precision = metric_df['precision'].mean()
result_mrr = metric_df['mrr'].mean()
result_recall = metric_df['recall'].mean()
result_f1 = metric_df['f1'].mean()

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


# In[433]:


f = open("C:/hlocal/TFM/nodos_problemas", 'a')
f.write(str(result_one_hit) + '\t' + str(result_precision) + '\t' + str(result_mrr) + '\t' + str(result_recall) + '\t' +  str(result_f1) + '\n') 
f.close()

