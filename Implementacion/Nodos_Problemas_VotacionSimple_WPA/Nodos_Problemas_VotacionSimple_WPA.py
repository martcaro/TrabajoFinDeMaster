
# coding: utf-8

# In[58]:

import pandas as pd
import numpy as np
import itertools as it
import networkx as nx
# Documentacion de la libreria: http://networkx.readthedocs.io/en/networkx-1.11/


# In[59]:

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


# In[60]:

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


# In[61]:

def filterWeight(weightUmbral, linksToFilter):
    """
        Funcion que filtra los enlaces de un grafo, para que el peso sea mayor o igual al dado
    """
    
    result = [(x, y, z) for (x, y, z) in linksToFilter if z >= weightUmbral]
    
    return result
    


# In[62]:

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


# In[63]:

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


# In[64]:

print(training_set)


# In[65]:

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


# In[66]:


# creo los enlaces a partir de la informacion de los nodos
links = createLinks(grouped, nodes)
# ahora filtro el grafo para que los enlaces solo tengan el peso que quiero
linksFiltered = filterWeight(5, links)

print(len(linksFiltered))

# aqui creo el grafo 
graph = create_graph_nx(nodes, linksFiltered)


# In[67]:

def apply_wpa(row, graph):
    """
        Funcion que devuelve el valor de weighted preferential attachment
    """
    value_wpa = graph.degree(row["one"], weight="weight") * graph.degree(row["two"], weight="weight")
        
    return value_wpa

def create_wpa_data(graph, nodes):

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
    dataFrame_pa['wpa'] = dataFrame_pa.apply (lambda row: apply_wpa(row, graph), axis=1)


    return dataFrame_pa


# In[68]:

wpa_df = create_wpa_data(graph, nodes)
print(wpa_df)


# In[69]:

# creo un diccionario que va a tener a los usuarios como keys y a los problemas que ha hecho como valores
# a partir del conjunto de entrenamiento
grouped_user = training_set.groupby('user_id')['problem_id'].apply(list)

# convierto la serie en un dataframe
df_users = pd.DataFrame({'user_id':grouped_user.index, 'list_problem_id':grouped_user.values})

print(df_users)


# In[70]:

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
    
    


# In[71]:

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


# In[72]:

# ahora tengo que hacer el filtro en este dataframe, de forma que solo aparezcan las filas en las que len_training y 
# len_evaluation sea >=5
datraframe_user_filter = datraframe_user_filter[(datraframe_user_filter['len_training'] >= 5) & (datraframe_user_filter['len_evaluation'] >=5)]
print(datraframe_user_filter)

# aqui voy a guardar la lista de usuarios a los que voy a recomendar
user_list_to_recommend = sorted(datraframe_user_filter['user_id'].tolist())
print(user_list_to_recommend)
print(len(user_list_to_recommend))


# In[73]:

# ahora tengo que filtrar df_users para que solo contenga las filas en las que los usuarios
# pertenecen a la anterior lista

df_users = df_users[df_users['user_id'].isin(user_list_to_recommend)]
print(df_users)


# In[74]:

# creo el nuevo dataframe con los resultados 
column_problem_recomend = {'problem_id': nodes}
dataframe_problem_recomend = pd.DataFrame.from_dict(column_problem_recomend)

print(dataframe_problem_recomend)


# In[75]:

def getSimilarProblems(row, wpa_df):
    """
        Funcion que devuelve la lista de los problemas similares del problema de la fila: aquellos que tienen un valor wpa
        mayor a cero
    """
    # print(row['user_id'])
    
    # obtengo dos df con los usuarios que tienen usuarios vecinos con el usuario de la fila 
    column_result_one_tmp = wpa_df[wpa_df['one'] == row['problem_id']]
    column_result_one = column_result_one_tmp[column_result_one_tmp['wpa'] > 0]
    column_result_two_tmp = wpa_df[wpa_df['two'] == row['problem_id']]
    column_result_two = column_result_two_tmp[column_result_two_tmp['wpa'] > 0]
    
    # saco las listas de usuarios con usuarios comunes
    list_one = list(column_result_one['two'])
    list_two = list(column_result_two['one'])
    
    # la concateno sin tener en cuenta repeticiones, porque nunca va a haber
    list_wpa = list_one + list_two
    
    # print(list_wpa)
    
    if list_wpa == []: # sino tiene vecinos en comun, pongo toda la lista de nodos
        list_wpa = graph.nodes()
        list_wpa.remove(row['problem_id']) # y elimino el nodo que estoy mirando
    
    # hago el filtro de los k mejores
    return list_wpa


# In[76]:

dataframe_problem_recomend['neighbors'] = dataframe_problem_recomend.apply (lambda row: getSimilarProblems(row, wpa_df), axis=1)

# aqui tengo la lista de usuarios con sus k usuarios similares
print(dataframe_problem_recomend)


# In[77]:

print(grouped)
# convierto la serie en un dataframe
df_users_recommend = pd.DataFrame({'problem_id':grouped.index, 'list_user_id':grouped.values})
print(df_users_recommend)


# In[78]:

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
        
        # aqui hago el filtro para que no se incluyan los problemas que ya ha hecho el usuario
        list_problems = [x for x in lista_problemas_comprobar if x not in list_problems_user]
        
        # ahora concateno el resultado
        list_result = list_result + list_problems
        # print(list_problems)
        # print(list_result)
        # print("---------------")
    
    return list_result


# In[79]:

# ahora para cada problema, hacer una lista de los usuarios similares a los que han realizado ese problema, 
# que no sean usuarios que han realizado ya ese problema
dataframe_problem_recomend['list_users'] = dataframe_problem_recomend.apply (lambda row: getUsersFromSimilarProblems(row, df_users_recommend), axis=1)
print(dataframe_problem_recomend)


# In[80]:

# elimino la columna de los problemas vecinos, ya que no interesa
del dataframe_problem_recomend['neighbors']

print(dataframe_problem_recomend)


# In[81]:

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


# In[82]:


# tengo que contar el numero de apariciones de cada usuario en la lista de usuarios

# voy a sacar primero una lista sin repeticiones
# ahora voy a crear una nueva columna que contenga la lista de usuarios sin repeticiones
dataframe_problem_recomend['lista_users_unique'] = dataframe_problem_recomend.apply(lambda row: delRepetitions(row), axis=1)

print(dataframe_problem_recomend)


# In[83]:

# ahora voy a separar cada problema_user para hacer la cuenta
# creo un nuevo dataframe que agrupa por el problema con su posible usuario
df_separation = dataframe_problem_recomend.groupby(['problem_id']).lista_users_unique.apply(lambda x: pd.DataFrame(x.values[0])).reset_index().drop('level_1', axis = 1)

df_separation.columns = ['problem_id', 'possible_user']

print(df_separation)


# In[84]:

def getScore(row, dataframe_problem_recomend):
    """
        Funcion que devuelve el numero de apariciones del posible usuario en la lista de usuarios
    """
    # saco la lista de usuarios original, de donde voy a sacar la cuenta
    df_lista_usuarios_original = dataframe_problem_recomend[dataframe_problem_recomend['problem_id'] == row['problem_id']]
    lista_usuarios_original = list(df_lista_usuarios_original['list_users'])[0]
    
    # print(lista_problemas_original)
    
    # saco el problema a contar
    user = row['possible_user']
    
    #print(problem)
    
    return lista_usuarios_original.count(user)


# In[85]:

# para cada usuario a "recomendar" para cada problema miro el numero de apariciones en su lista

df_separation['score'] = df_separation.apply (lambda row: getScore(row, dataframe_problem_recomend), axis=1)

print(df_separation)


# In[86]:

# ahora voy a ordenar en funcion del score de mayor a menor para cada usuario
# ahora lo que quiero es ordenar los problemas por cada usuario en funcion de su ponderacion
# primero ordeno por su valor de user y luego por el de ponderacion, de forma que quedan ordenador por su valor de ponderacion
df_separation = df_separation.sort_values(by=['possible_user', 'score'], ascending=False)
print(df_separation)


# In[87]:

print(df_separation[df_separation['score'] > 1])


# In[88]:

# ahora tengo que hacer un nuevo dataframe con usuario y problema, y una lista de recommendation 

# hago primero la agrupacion por usuario
grouped_r = df_separation.groupby('possible_user')

# hago la agregacion en una lista 
df_recommend = grouped_r.aggregate(lambda x:list(x))

#print(df_recommend)

# vuelvo a crear la estructura de los datos para poder trabajar con ellos
df_recommend = pd.DataFrame({'user_id':df_recommend.index.values, 'problem_id':df_recommend['problem_id'].tolist()})
print(df_recommend)


# In[89]:

def getProblems(row, df_recommend):
    """
        Funcion que copia los posibles problemas a recomendar para los usuarios a los que tengo que recomendar
    """
    
    # saco la lista de problemas 
    df_lista_problemas_original = df_recommend[df_recommend['user_id'] == row['user_id']]
    lista_problemas_original = list(df_lista_problemas_original['problem_id'])[0]
    
    return lista_problemas_original
    


# In[90]:

# df_users # en este dataframe estan los usuarios a los que quiero recomendar

# ahora voy a copiar los problemas para los usuarios a los que tengo que recomendar
df_users['recommendation'] = df_users.apply (lambda row: getProblems(row, df_recommend), axis=1)

print(df_users)


# In[91]:

# elimino la columna de los problemas, ya que no interesa
del df_users['list_problem_id']

print(df_users)


# In[92]:

def getKrecomFinal(row, k):
    """
        Funcion que saca las k mejores recomendaciones para el usuario
        Lo que hace es coger los primeros k valores de la lista de recomendaciones
    """
    return row['recommendation'][:k]


# In[93]:

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


# In[94]:

# elimino las columnas que no me interesan
del df_users['recommendation']

print(df_users)


# In[95]:

# ahora tengo que filtrar df_users_eval para que solo contenga las filas de los usuarios a los que hay que recomendar

df_users_eval_filter = df_users_eval[df_users_eval['user_id'].isin(user_list_to_recommend)]
print(df_users_eval_filter)


# In[96]:

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


# In[97]:

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


# In[98]:

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


# In[99]:

def precision(row, i):
    """
        Funcion que va a implementar la metrica precision en k: 
        (cuantos de los realizados por el usuario estan entre los recomendados) / todos los recomendados
    """
    name_column = 'recom_problems_' + str(i)
    num_problems_common = np.intersect1d(row[name_column], row['eval_problems'])
    
    # print(num_problems_common)
    
    return (len(num_problems_common)/len(row[name_column]))


# In[100]:

def recall(row, i):
    """
        Funcion que implementa la metrica recall
        (cuantos de los realizados por el usuario estan entre los recomendados) / todos los evaluados
    """
    name_column = 'recom_problems_' + str(i)
    num_problems_common = np.intersect1d(row[name_column], row['eval_problems'])
    
    # print(num_problems_common)
    
    return (len(num_problems_common)/len(row['eval_problems']))


# In[101]:

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


# In[102]:

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


# In[103]:



f = open("C:/hlocal/TFM/vot_simple_wpa", 'a')

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
    

