
# coding: utf-8

# In[460]:

import pandas as pd
import numpy as np
import itertools as it
import networkx as nx
# Documentacion de la libreria: http://networkx.readthedocs.io/en/networkx-1.11/


# In[461]:

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


# In[462]:

# MAIN
# ---------

# se guarda en la variable df (DataFrame) toda la base de datos
df = pd.read_csv('bbdd_orderbydate.csv')

# aqui quito los problemas que no existian despues de la fecha umbral
df = df[df['problem_id'] <= 511] 

# construyo el conjunto de entrenamiento
training_set = filterData(df, True, "2016-10-21 00:00:00")

print(training_set)

# obtengo los nodos del grafo, esta vez los nodos son los usuarios y no los problemas:
nodes = training_set.user_id.unique()

# creo un diccionario que va a tener a los usuarios como keys y los valores seran los
# problemas que ha hecho ese usuario
grouped = training_set.groupby('user_id')['problem_id'].apply(list)

# muestra el numero de usuarios
print(len(nodes))

# muestra la lista de problemas que ha hecho cada usuario
print(grouped)


# In[463]:

print(training_set)


# In[464]:

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


# In[465]:

# In[3]:

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
    
    # hago todas las posibles combinaciones de usuarios
    for fst, snd in it.combinations(nodos, 2):
        # obtengo el peso pasando la lista de problemas que ha hecho cada usuario
        peso = compareNodes(prob_us_set[fst], prob_us_set[snd])
        if peso >= 1:
            resultado.append((fst, snd, peso))
            
            
            
    return resultado


# In[4]:

def filterWeight(weightUmbral, linksToFilter):
    """
        Funcion que filtra los enlaces de un grafo, para que el peso sea mayor o igual al dado
    """
    
    result = [(x, y, z) for (x, y, z) in linksToFilter if z >= weightUmbral]
    
    return result
    


# In[5]:

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


# In[466]:

# creo los enlaces a partir de la informacion de los nodos
links = createLinks(grouped, nodes)
# ahora filtro el grafo para que los enlaces solo tengan el peso que quiero
linksFiltered = filterWeight(5, links)

print(len(linksFiltered))

# aqui creo el grafo 
graph = create_graph_nx(nodes, linksFiltered)


# In[467]:

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
    


# In[468]:

# aqui voy a hacer el filtro de usuarios de forma que para hacer las recomendaciones solo tengamos en 
# cuenta aquellos usuarios que han hecho 5 o mas problemas tanto antes de la fecha limite como despues

# convierto la serie en un dataframe
df_users = pd.DataFrame({'user_id':grouped.index, 'list_problem_id':grouped.values})

# primero guardo la lista de usuarios
user_list = df.user_id.unique()

# la meto en un dataframe 
column_user_filter = {'user_id': user_list}
datraframe_user_filter = pd.DataFrame.from_dict(column_user_filter)


# ahora tengo que calcular para cada fila, el numero de problemas que han hecho en el training_set, evaluation_set
datraframe_user_filter['len_training'] = datraframe_user_filter.apply (lambda row: lenProblemsDone(row, df_users), axis=1)
datraframe_user_filter['len_evaluation'] = datraframe_user_filter.apply (lambda row: lenProblemsDone(row, df_users_eval), axis=1)
print(datraframe_user_filter)


# In[469]:

# ahora tengo que hacer el filtro en este dataframe, de forma que solo aparezcan las filas en las que len_training y 
# len_evaluation sea >=5
datraframe_user_filter = datraframe_user_filter[(datraframe_user_filter['len_training'] >= 5) & (datraframe_user_filter['len_evaluation'] >=5)]
print(datraframe_user_filter)

# aqui voy a guardar la lista de usuarios a los que voy a recomendar
user_list_to_recommend = sorted(datraframe_user_filter['user_id'].tolist())
print(user_list_to_recommend)
print(len(user_list_to_recommend))


# In[470]:

# ahora tengo que filtrar df_users para que solo contenga las filas en las que los usuarios
# pertenecen a la anterior lista

df_users_recommend = df_users[df_users['user_id'].isin(user_list_to_recommend)]
print(df_users_recommend)


# In[471]:


# en df_new tengo los usuarios a los que tengo que hacer recomendaciones

# primero guardo la lista de usuarios
user_list_recomend = df_users_recommend.user_id.unique()

# creo el nuevo dataframe con los resultados 
column_user_recomend = {'user_id': user_list_to_recommend}
dataframe_user_recomend = pd.DataFrame.from_dict(column_user_recomend)

print(dataframe_user_recomend)


# In[472]:

def apply_pa(row, graph):
    """
        Funcion que devuelve el valor de preferential attachment
    """
    values_pa = nx.preferential_attachment(graph, [(row['one'], row['two'])])
    
    value_pa = 0
    for u, v, p in values_pa:
        value_pa = p # saco el valor
        
    return value_pa

def create_pa_data(graph, nodes):

    # Ahora voy a construir un DataFrame que tenga dos columnas con todas las posibles combinaciones de problemas, y otra 
    # columna con el valor de pa para ese par de problemas
    fst_column = list()
    snd_column = list()
    for fst, snd in it.combinations(nodes, 2):
        fst_column.append(fst)
        snd_column.append(snd)

    d = {'one' : fst_column,
        'two' : snd_column}
    dataFrame_pa = pd.DataFrame(d)


    # Aplico la funcion a cada fila
    dataFrame_pa['pa'] = dataFrame_pa.apply (lambda row: apply_pa(row, graph), axis=1)


    return dataFrame_pa


# In[473]:

pa_df = create_pa_data(graph, nodes)
print(pa_df)


# In[474]:

def getCommonNeighbors(row, pa_df):
    """
        Funcion que devuelve la lista de los usuarios de ese usuario que tienen vecinos en comun
    """
    # print(row['user_id'])
    
    # obtengo dos df con los usuarios que tienen usuarios vecinos con el usuario de la fila 
    column_result_one_tmp = pa_df[pa_df['one'] == row['user_id']]
    column_result_one = column_result_one_tmp[column_result_one_tmp['pa'] > 0]
    column_result_two_tmp = pa_df[pa_df['two'] == row['user_id']]
    column_result_two = column_result_two_tmp[column_result_two_tmp['pa'] > 0]
    
    # saco las listas de usuarios con usuarios comunes
    list_one = list(column_result_one['two'])
    list_two = list(column_result_two['one'])
    
    # la concateno sin tener en cuenta repeticiones, porque nunca va a haber
    list_pa = list_one + list_two
    
    # print(list_pa)
    
    if list_pa == []: # sino tiene vecinos en comun, pongo toda la lista de nodos
        list_pa = graph.nodes()
        list_pa.remove(row['user_id']) # y elimino el nodo que estoy mirando
    
    # hago el filtro de los k mejores
    return list_pa


# In[475]:

dataframe_user_recomend['neighbors'] = dataframe_user_recomend.apply (lambda row: getCommonNeighbors(row, pa_df), axis=1)

# aqui tengo la lista de usuarios con sus k usuarios similares
print(dataframe_user_recomend)


# In[476]:

# ahora voy a separar cada user-problema_a_recomendar para hacer la cuenta
# creo un nuevo dataframe que agrupa por el primer problema y tiene su posible recomendacion
df_separation = dataframe_user_recomend.groupby(['user_id']).neighbors.apply(lambda x: pd.DataFrame(x.values[0])).reset_index().drop('level_1', axis = 1)

df_separation.columns = ['user_id', 'neighbors']

print(df_separation)


# In[477]:

def getProblemsFromSimilarUSers(row, df_users, df_users_recommend):
    """
        Funcion que va a devolver por cada fila una lista procedente de problemas que han
        hecho los usuarios similares a ese. Además eliminara los problemas que ya haya hecho el usuario
    """
    
    # obtengo la lista de problemas que ha hecho el usuario en cuestion
    list_problems_users = df_users_recommend[df_users_recommend['user_id'] == row['user_id']]
    list_problems_user = list(list_problems_users['list_problem_id'])[0]
    
    # aqui saco la lista de problemas que ha hecho el usuario similar
    list_problems_df = df_users[df_users['user_id'] == row['neighbors']]    
    lista_problemas_comprobar = list(list_problems_df['list_problem_id'])[0]
    
    

    # aqui hago el filtro para que no se incluyan los problemas que ya ha hecho el usuario
    list_problems = [x for x in lista_problemas_comprobar if x not in list_problems_user]
    
    return list_problems


# In[478]:

# ahora para cada lista de de usuarios, hacer una lista de los problemas realizados por esos usuarios, 
# que no los haya realizado ya el usuario
df_separation['list_problems'] = df_separation.apply (lambda row: getProblemsFromSimilarUSers(row, df_users, df_users_recommend), axis=1)

print(df_separation)


# In[479]:

def pa_value(one, two, graph):
    """
        Funcion que devuelve el numero de vecinos en comun de esos dos nodos
    """
    values_pa = nx.preferential_attachment(graph, [(one, two)])
    
    value_pa = 0
    for u, v, p in values_pa:
        value_pa = p # saco el valor
        
    return value_pa


# In[480]:

def getValueSimilarMetric(row, df_users_recommend, graph):
    """
        Funcion que va a devolver por cada fila el valor de similaridad entre los dos usuarios de esa fila
    """
    
    return pa_value(row['user_id'], row['neighbors'], graph)


# In[481]:

# ahora para cada lista de de usuarios, hacer una lista de los problemas realizados por esos usuarios, 
# que no los haya realizado ya el usuario
df_separation['sim_value'] = df_separation.apply (lambda row: getValueSimilarMetric(row, df_users_recommend, graph), axis=1)

print(df_separation)


# In[482]:

# ahora voy a borrar la columna neighbors
del df_separation['neighbors']


# In[483]:

df_separation


# In[484]:

# ahora voy a ordenar por el valor de pa agrupando por el usuario
df_separation = df_separation.sort_values(by=['user_id', 'sim_value'], ascending=False)
print(df_separation)


# In[485]:

df_separation[df_separation['user_id'] == 1619]


# In[486]:

# ahora voy a borrar la columna de similitud por que ya no me hace falta
del df_separation['sim_value']


# In[487]:

df_separation


# In[488]:

# hago primero la agrupacion por usuario
grouped_r = df_separation.groupby('user_id')

# hago la agregacion en una lista 
df_recommend_final = grouped_r.aggregate(lambda x:list(x))

print(df_recommend_final)


# In[489]:

def concatenateLists(l):
    """
        Funcion auxiliar para concatenar listas que estan dentro de una lista
    """
    size = len(l)
    
    result = list()
    
    for i in range(0, size):
        value = l[i]
        result = result + value
    
    return result


# In[490]:

def concatenateListsRecom(row):
    """
        Funcion que crea una lista de la concatenacion de listas
    """
    
    value = concatenateLists(row['list_problems'])
    
    return value


# In[491]:

# ahora para cada lista de de usuarios, hacer una lista de los problemas realizados por esos usuarios, 
# que no los haya realizado ya el usuario
df_recommend_final['recommendation'] = df_recommend_final.apply (lambda row: concatenateListsRecom(row), axis=1)

print(df_recommend_final)


# In[492]:

# ahora voy a borrar la columna de similitud por que ya no me hace falta
del df_recommend_final['list_problems']


# In[493]:

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


# In[494]:

# ahora voy a eliminar las repeticiones
df_recommend_final['recommendation_unique'] = df_recommend_final.apply(lambda row: delRepetitions(row), axis=1)

print(df_recommend_final)


# In[495]:

del df_recommend_final['recommendation']


# In[496]:

df_recommend_final


# In[497]:

def getKrecomFinal(row, k):
    """
        Funcion que saca las k mejores recomendaciones para el usuario
        Lo que hace es coger los primeros k valores de la lista de recomendaciones
    """
    if k == 1:
        value = list()
        value.append(row['recommendation_unique'][:k])
        return value
    else:
        return row['recommendation_unique'][:k]


# In[498]:

k = 10
# ahora saco los k mejores problemas para cada usuario
df_recommend_final['k_recommendation'] = df_recommend_final.apply(lambda row: getKrecomFinal(row, k), axis=1)

print(df_recommend_final)


# In[499]:

del df_recommend_final['recommendation_unique']


# In[500]:

df_recommend_final


# In[501]:

# ahora tengo que filtrar df_users_eval para que solo contenga las filas de los usuarios a los que hay que recomendar

df_users_eval_filter = df_users_eval[df_users_eval['user_id'].isin(user_list_to_recommend)]
print(df_users_eval_filter)


# In[502]:

list_eval_problems = df_users_eval_filter['list_problem_id'].tolist()
list_recom_problems = df_recommend_final['k_recommendation'].tolist()


# meto toda la informacion en un dataframe para obtener las metricas
set_df_metric = {'user_id': user_list_to_recommend, 'eval_problems': list_eval_problems, 'recom_problems': list_recom_problems}
metric_df = pd.DataFrame.from_dict(set_df_metric)

print(metric_df)


# In[503]:

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


# In[504]:

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


# In[505]:

def precision(row):
    """
        Funcion que va a implementar la metrica precision en k: 
        (cuantos de los realizados por el usuario estan entre los recomendados) / todos los recomendados
    """
    
    num_problems_common = np.intersect1d(row['recom_problems'], row['eval_problems'])
    
    # print(num_problems_common)
    
    return (len(num_problems_common)/len(row['recom_problems']))


# In[506]:

def recall(row):
    """
        Funcion que implementa la metrica recall
        (cuantos de los realizados por el usuario estan entre los recomendados) / todos los evaluados
    """
    num_problems_common = np.intersect1d(row['recom_problems'], row['eval_problems'])
    
    # print(num_problems_common)
    
    return (len(num_problems_common)/len(row['eval_problems']))


# In[507]:

def f1(row):
    """
        Funcion que calcula el f1 en funcion de precision y recall
    """
    denominador = row['precision'] + row['recall']
    
    if denominador == 0:
        return 0
    else:
        return (2 * row['precision'] * row['recall']) / denominador


# In[508]:

# ahora voy a calcular una metrica para cada usuario
metric_df['one_hit'] = metric_df.apply(lambda row: one_hit(row), axis=1)
metric_df['mrr'] = metric_df.apply(lambda row: mrr(row), axis=1)
metric_df['precision'] = metric_df.apply(lambda row: precision(row), axis=1)
metric_df['recall'] = metric_df.apply(lambda row: recall(row), axis=1)
metric_df['f1'] = metric_df.apply(lambda row: f1(row), axis=1)
print(metric_df)


# In[509]:

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


# In[510]:


f = open("C:/hlocal/TFM/nodos_usuarios2", 'a')
f.write(str(result_one_hit) + '\t' + str(result_precision) + '\t' + str(result_mrr) + '\t' + str(result_recall) + '\t' +  str(result_f1) + '\n') 
f.close()

