import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#import numpy as np

from numpy import sum,array,median,abs,nan,ndarray,concatenate,mean,max,var,std,isnan

#import pandas as pd
from scipy.signal import savgol_filter
import datetime
from scipy.stats import median_absolute_deviation
import ast
import logging
import json

logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s')    



def interp(js):
    return [json.loads(js['data']['values']),ast.literal_eval(js['data']['datetime'])]


    

def mean_squared_error(ypred,yreal):

    if len(ypred) == len(yreal):
        MSE = sum((ypred - yreal)**2)/len(ypred)
        
    else:
        MSE = 0
        logging.error('AI_warning_4: prediccion y datos reales tienen distinta logitud MSE fijado a 0')
    return MSE
     


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = array(y_true), array(y_pred)
    return mean(abs((y_true - y_pred) / y_true)) * 100

    
def savgol(X,n):
    '''
    Funcion de suavizado de la senal.
    
    Parameters
    ----------
    
    X : array
        array de datos que se desea suavizar
    n : int
        numero de datos que se emplearan para el suavizado
    '''
    
    savgol_data = savgol_filter(X, n, 1)[:-int(n/3)]
    
    return savgol_data
    


def mad_method(df):
    
    '''
    Funcion que encuentra los outliers de una serie de valores
    
    Attributes
    ----------
    
    df : array
        array al que se le van a encontrar los outliers
    
    '''
    df = array(df)
    med = median(df)
    mad = abs(median_absolute_deviation(df))
    threshold = 3
    outlier = []
    index=0    
    for i, v in enumerate(df):
        t = (v-med)/mad
        if t > threshold:
            outlier.append(i)
        else:
            continue
    return outlier


def pipeline(data = None,sav = 111,N = 200,n= 40):
    
    '''
    Funcion que preprocesa los datos a los que se va a realizar la prediccion. 
    El proceso es el siguiente: 
        se encuentran los outliers de los datos, estos se etiquetan como datos ausentes 
        estos datos se interpolan con el objetivo de eliminar estos valores ausentes
        se suavizan los datos mediante un filtro.
    
    Attributes
    ----------
    
    data : array 
        datos de input a preprocesar
    
    real : array
        datos con los que se realizara la validacion del modelo
    
    sav : int 
        grado de suavizado de los datos
    
    '''
    header = data['header']
    
    data = interp(data)
    
    if data is not None and type(data) == list :
        
        data[0] = data[0]
        data[1] = data[1]
        #se ajusta el grado a grado de suavizado para que sea impar
        if sav%2 == 0: sav += 1

        #deteccion de ouliers
        data[0] = array(data[0],dtype = float)
        comp = float(sum(~isnan(data[0]))/len(data[0]))
        index_outlier = []

        data[0][isnan(data[0])] = 0   


        data_int = savgol(data[0],sav)
        
        
        if len(data_int) < N:#ventana de datos adecuada para el modelo  
            logging.error('AI_error_0: no hay sufientes datos para realizar la prediccion. Minimo numero de datos: 200/' + str(len(data_int)))

        else:
            try:
                #se aplica el filtro
                data_filter = data_int.reshape(-1,1).reshape(1,2*N,1)

                #se prueba 
                if len(data_int) >= N:
                    
                    
                    pre = data_int[-(2*N):-N].reshape(-1,1).reshape(1,N,1)
                else:
                    pre = None
                    logging.warning('AI_warning_0: no hay suficientes datos para validacion. Minimo numero de datos: 200/' + str(len(data_int)))
                    # print("not enough data for validation")

                return [array(data_filter,dtype = float),comp,data[1],len(index_outlier),array(pre,dtype = float),array(data[0],dtype = float),sav,array(data_int,dtype = float),header]

            except Exception as e: 
                print(e)
                return [e]

    else:
        if data is None: logging.error('AI_error_1: no hay datos de entrada')
        if not isinstance(data,list): logging.error('AI_error_2: formato de datos no adecuado')
        if not isinstance(data[0],ndarray): logging.error('AI_error_3: lista de inputs mal cosntruida')
        if not isinstance(data[1],list): logging.error('AI_error_3: lista de inputs mal construida')
        if not isinstance(data[0][0], float): logging.error('AI_error_4: tipo de dato no adecuado')
        if not isinstance(data[1][0], datetime.datetime): logging.error('AI_error_4: tipo de dato no adecuado')
        
    

            
    

    
    
