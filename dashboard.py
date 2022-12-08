import pandas as pd
import streamlit as st
import requests
import seaborn as sns
import ast
from streamlit_echarts import st_echarts
from st_aggrid import AgGrid
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")




def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}
    data_json = {'index': data}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


def main():
    FastAPI_URI_pred = 'http://127.0.0.1:8000/predict'
    FastAPI_URI_show = 'http://127.0.0.1:8000/show'
    FastAPI_URI_explain = 'http://127.0.0.1:8000/explain'

    api_choice = st.sidebar.selectbox(
        'Quelle API souhaitez vous utiliser',
        ['FastAPI'])

    st.title('Prédiction du remboursement d un client')
    
    index = st.selectbox('Quel client souhaitez vous consulter ?', [0,1,2])
   # revenu_med = st.number_input('Revenu médian dans le secteur (en 10K de dollars)',
  #                               min_value=0., value=3.87, step=1.)
#
  #  age_med = st.number_input('Âge médian des maisons dans le secteur',
   #                           min_value=0., value=28., step=1.)
#
  #  nb_piece_med = st.number_input('Nombre moyen de pièces',
  #                                 min_value=0., value=5., step=1.)
#
#  "  nb_chambre_moy = st.number_input('Nombre moyen de chambres',
  #                                   min_value=0., value=1., step=1.)
#
 #   taille_pop = st.number_input('Taille de la population dans le secteur',
# " "                               min_value=0, value=1425, step=100)
#"
#    occupation_moy = st.number_input('Occupation moyenne de la maison (en nombre d\'habitants)',
 #                                    min_value=0., value=3., step=1.)
#
  #  latitude = st.number_input('Latitude du secteur',
 #                              value=35., step=1.)
#
 #   longitude = st.number_input('Longitude du secteur',
 #                               value=-119., step=1.)
#"
    col1, col2 = st.columns(2)
    with col1:
        show_btn = st.button('Afficher Information du client')

        if show_btn:

            data = index
            Affichage = None

            if api_choice == 'FastAPI':
                Affichage = request_prediction(FastAPI_URI_show, data)
            AgGrid(pd.DataFrame(ast.literal_eval(Affichage)))
            #st.write(f'{Affichage}')
        
        
    
    
    with col2:    
        predict_btn = st.button('Prédire')
        if predict_btn:
            data = index
            pred = None

            if api_choice == 'FastAPI':
                pred = request_prediction(FastAPI_URI_pred, data)
            st.write(
                f'La probabilité de défaut de ce client est de {pred}')

        
    

    
    explain_btn = st.button('Expliquer')
    if explain_btn:
        data = index
        explain_plot = None

        if api_choice == 'FastAPI':
            explain_plot = ast.literal_eval(request_prediction(FastAPI_URI_explain, data))
        df_explain_plot = pd.DataFrame(explain_plot)
       # st.write(df_explain_plot)
        df_explain_plot['positive'] = df_explain_plot[1] > 0
        fig = sns.barplot(data = df_explain_plot, x = 1, y = 0, hue = 'positive' , palette = 'rocket').get_figure()
        plt.xlabel('Pouvoir prédictif de la variable')
        plt.ylabel('Variables')
        st.pyplot(fig=fig)

    test_btn = st.button('tester')    
        
    if test_btn:
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
                                [0.3, '#2FEB00'],
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
if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    
    
    
    
    
    