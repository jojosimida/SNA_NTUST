import numpy as np
import pandas as pd
from sklearn.svm import SVC
import networkx as nx

data_path = '/notebooks/social network/HW1/data/'
p1 = 'Period1.csv'
p2 = 'Period2.csv'
test = 'TestData.csv'

df = pd.read_csv('process1.csv')
X = df.iloc[:,2:-2]
Y = df.iloc[:,-1]

clf = SVC()
clf.fit(X, Y)

test_df = pd.read_csv(data_path+test)

p1_df = pd.read_csv(data_path+p1)
p1_target = set(p1_df['target id'].values.tolist())
p2_df = pd.read_csv(data_path+p2)
p2_target = set(p2_df['target id'].values.tolist())
all_target = list(p1_target|p2_target)

def to_graph(df):
    G=nx.Graph()
    for i in range(df.shape[0]):
        G.add_edge(df.iloc[i]['source id'],df.iloc[i]['target id'])
        
    return G

p1_nonedge = df[df['label']==0]

p1_nonedge_G = to_graph(p1_nonedge)
true_p1_edges = list(nx.non_edges(p1_nonedge_G))
p2_edge = df[df['label']==1]

p2_edge_G = to_graph(p2_edge)
new_network_edge = nx.complement(p2_edge_G)

new_network_edge.add_edges_from(true_p1_edges)

def compute_common_neigh(G, nonedge):
    common_neigh = [(e[0], e[1], len(list(nx.common_neighbors(G,e[0],e[1])))) for e in nonedge]
    return common_neigh


def get_list_value(feature_list):
    v = []
    for i in feature_list:
        v.append(i[2])
        
    to_value = pd.Series(v).values
    
    return to_value


def print_output(out):
    print(len(out))
    print(out[:5])

def get_feature(nonedge, G, df):
    #=========common_neigh===========
    common_neigh = compute_common_neigh(G, nonedge)
    v = get_list_value(common_neigh)
    df['common_neigh'] = v
    print_output(common_neigh)
    
    #=========jaccard_coefficient===========
      
    jaccard = list(nx.jaccard_coefficient(G, nonedge))
    v = get_list_value(jaccard)
    df['jaccard'] = v    
    print_output(jaccard)
    
    resource_alloc = list(nx.resource_allocation_index(G, nonedge))
    v = get_list_value(resource_alloc)
    df['resource_alloc'] = v      
    print_output(resource_alloc)
    
    adamic_adar = list(nx.adamic_adar_index(G, nonedge))
    v = get_list_value(adamic_adar)
    df['adamic_adar'] = v       
    print_output(adamic_adar)
    
    pref_attach = list(nx.preferential_attachment(G, nonedge))
    v = get_list_value(pref_attach)
    df['pref_attach'] = v      
    print_output(pref_attach)


test_df = test_df.drop(columns=['year'])

def get_row_data(df):
    edge_list = []
    for index, row in df.iterrows():
        edge_list.append((row['source id'],row['target id']))
        
    return edge_list

edge_list = get_row_data(test_df)
node_list = [ int(i)  for i in list(new_network_edge.nodes())]

label_list = []
cc = 0
for i,v in enumerate(edge_list):
    if (v[0] not in node_list) or (v[1] not in node_list):
        if v[1] in all_target:
            edge_list[i] = (9601024, 6005)
            label_list.append(1)

        else:
            edge_list[i] = (9601024, 9306120)
            label_list.append(0)
    else:
        label_list.append(1)
        cc+=1
        
    if (i+1)%1000==0:    
        print(str(i+1)+' done')

get_feature(edge_list, new_network_edge, test_df)

test_X = test_df.iloc[:,2:-1]
pre = clf.predict(test_X)

col = ['target id', 'label']
target = list(range(1,test_df.shape[0]+1))

df = pd.DataFrame({'target id':target,
                   'label':pre
                  },columns=col)

df.to_csv('sub_toogod.csv', index=False)