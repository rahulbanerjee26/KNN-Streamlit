import streamlit as st
import pandas as pd
from sklearn.datasets import make_blobs
import plotly.express as px
import plotly.graph_objects as go


def min_max_normalize(lst):
  minimum = min(lst)
  maximum = max(lst)
  normalized = [(val - minimum)/(maximum - minimum) for val in lst]
  return normalized

def distance(element1 , element2):
    x_distance = (element1[0] - element2[0])**2
    y_distance = (element1[1] - element2[1])**2
    return (x_distance + y_distance)**0.5

def get_label(neighbours, y):
    zero_count , one_count = 0,0
    for element in neighbours:
        if y[element[1]] == 0:
            zero_count +=1
        elif y[element[1]] == 1:
            one_count +=1
    if zero_count == one_count:
        return y[neighbours[0][1]]
    return 1 if one_count > zero_count else 0


def find_nearest(x , y , input , k):
    distances = []
    for id,element in enumerate(x):
        distances.append([distance(input , element),id])
    distances = sorted(distances)
    predicted_label = get_label(distances[0:k] , y)
    return predicted_label, distances[0:k] , distances[k:]


st.title("KNN Visualize")

x , y = make_blobs(n_samples = 100 , n_features = 2 , centers = 2, random_state= 2)



x_input = st.slider("Choose X input", min_value=0.0, max_value=1.0,key='x')
y_input = st.slider("Choose Y input", min_value=0.0, max_value=1.0,key='y')
k = st.slider("Choose value of K", min_value=1, max_value=10,key='k')

input = (x_input,y_input)

# Normalizing Data
x[:,0] = min_max_normalize(x[:,0])
x[:,1] = min_max_normalize(x[:,1])

# Dataframe
df = pd.DataFrame(x , columns = ['Feature1' , 'Feature2'] )
df['Label'] = y

# Initial Data Plot
st.dataframe(df)
fig = px.scatter(df, x = 'Feature1' , y='Feature2', symbol='Label',symbol_map={'0':'square-dot' , '1':'circle'})
fig.add_trace(
    go.Scatter(x= [input[0]], y=[input[1]], name = "Point to Classify", )
)
st.plotly_chart(fig)

#Finding Nearest Neighbours
predicted_label , nearest_neighbours, far_neighbours = find_nearest(x ,y , input ,k)

st.title('Prediction')
st.subheader('Predicted Label : {}'.format(predicted_label))

nearest_neighbours = [[neighbour[1],x[neighbour[1],0],x[neighbour[1],1],neighbour[0],y[neighbour[1]]] for neighbour in nearest_neighbours]

nearest_neighbours = pd.DataFrame(nearest_neighbours , columns = ['id','Feature1','Feature2','Distance','Label'])
st.dataframe(nearest_neighbours)

far_neighbours = [[neighbour[1],x[neighbour[1],0],x[neighbour[1],1],neighbour[0],y[neighbour[1]]] for neighbour in far_neighbours]
far_neighbours = pd.DataFrame(far_neighbours , columns = ['id','Feature1','Feature2','Distance','Label'])

fig2 = px.scatter(far_neighbours,x='Feature1',y='Feature2',symbol='Label',symbol_map={'0':'square-dot' , '1':'circle'})

for index,neighbour in nearest_neighbours.iterrows():
    fig2.add_trace(
        go.Scatter( x=[input[0], neighbour['Feature1']], y=[input[1], neighbour['Feature2']],mode='lines+markers' , name = 'id {}'.format(int(neighbour['id'])) )
    )

st.plotly_chart(fig2)