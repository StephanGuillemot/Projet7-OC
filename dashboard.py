import pandas as pd
import streamlit as st
import requests
import seaborn as sns
import ast
from streamlit_echarts import st_echarts
from st_aggrid import AgGrid
import matplotlib.pyplot as plt

import pickle
# streamlit run dashboard.py

st.set_page_config(layout="wide")

# Import data
data_test = open("data_test.pkl","rb")
data_test = pickle.load(data_test)


def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}
    data_json = {'id_cli': data}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


def main():
    """
    Configure les differents ports de l'API
    Cree une liste deroulante pour chosir l'application utilise
    """
    FastAPI_URI_pred = 'https://projet7stephanguillemot.herokuapp.com/predict'
    FastAPI_URI_show = 'https://projet7stephanguillemot.herokuapp.com/show'
    FastAPI_URI_explain = 'https://projet7stephanguillemot.herokuapp.com/explain'

    st.title('Prédiction du remboursement d un client')
    
    index = st.selectbox('Quel client souhaitez vous consulter ?', data_test['SK_ID_CURR'].iloc[:100].to_list())
    #index = st.selectbox('Quel client souhaitez vous consulter ?', [387758])
    
    #col1, col2 = st.columns(2)
    with st.container():
       # show_btn = st.button('Afficher Information du client')

       # if show_btn:

        data = index
       # Affichage = None

        Affichage = request_prediction(FastAPI_URI_show,  data)
        AgGrid(pd.DataFrame(ast.literal_eval(Affichage)))
        #st.write(f'{Affichage}')
        
        
    
    
    with st.container():    
      #  predict_btn = st.button('Prédire')
       # if predict_btn:
        data = index
        pred = request_prediction(FastAPI_URI_pred, data)



        option = {
            "tooltip": {
                "formatter": '{a} <br/>{b} : {c}%'
            },
            "series": [{
                "name": 'Proba',
                "type": 'gauge',
                "axisLine": {
                    "lineStyle": {
                        "width": 5,
                        "color": [
                                [0.3, '#2FEB00'], # seuils intermédiaires
                                [0.7, '#FF8114'],
                                [1, '#FF0000']
                            ]
                        },},
                "min" : 0,
                "max" : 100,
                "startAngle": 180,
                "endAngle": 0,
                "progress": {
                    "show": "true"
                },
                "radius":'100%', 

                "itemStyle": {
                    "color": '#58D9F9',
                    "shadowColor": 'rgba(0,138,255,0.45)',
                    "shadowBlur": 10,
                    "shadowOffsetX": 2,
                    "shadowOffsetY": 2,
                    "radius": '55%',
                },
                "progress": {
                    "show": "true",
                    "roundCap": "true",
                    "width": 6
                },
                "pointer": {
                    "length": '60%',
                    "width": 8,
                    "offsetCenter": [0, '5%']
                },
                "detail": {
                    "valueAnimation": "true",
                    "formatter": '{value}',
                    "backgroundColor": '#58D9F9',
                    "borderColor": '#999',
                    "borderWidth": 4,
                    "width": '60%',
                    "lineHeight": 20,
                    "height": 20,
                    "borderRadius": 188,
                    "offsetCenter": [0, '40%'],
                    "valueAnimation": "true",
                },
                "data": [{
                    "value": round(float(pred['prediction']) * 100, 1) ,
                    #"value": 40,
                    "name": 'Prediction'
                }]
            }]
        };


        st_echarts(options=option, key="1") 

        
    

    
    #explain_btn = st.button('Expliquer')
   # if explain_btn:
    with st.container():
        data = index

        explain_plot = ast.literal_eval(request_prediction(FastAPI_URI_explain, data))
        df_explain_plot = pd.DataFrame(explain_plot)
       # st.write(df_explain_plot)
        df_explain_plot['positive'] = df_explain_plot[1] > 0
        fig = sns.barplot(data = df_explain_plot, x = 1, y = 0, hue = 'positive' , palette = 'rocket').get_figure()
        plt.xlabel('Pouvoir prédictif de la variable')
        plt.ylabel('Variables')
        st.pyplot(fig=fig)
        
        
    with st.container() :
        Colonne = st.selectbox('Quel variable souhaitait vous consulter ? ', data_test.drop(['SK_ID_CURR'], axis = 1).columns)
        fig = sns.kdeplot(data = data_test, x = Colonne, hue = 'TARGET', common_norm = True)
        plt.plot([0.5, float(data_test[data_test['SK_ID_CURR'] == index][Colonne])] , [0,1], 'r', linestyle = 'dashed')
        plt.xlabel(f'Valeur {Colonne}')
        plt.ylabel(f'Répartition {Colonne} Defaut et Sains')
        st.pyplot(fig=fig)
        
        
        
if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    
    
    
    
    
    