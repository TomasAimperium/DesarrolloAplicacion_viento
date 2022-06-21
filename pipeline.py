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
# from sklearn.metrics import mean_squared_error
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
    
    savgol_data = savgol_filter(X, n, 1)
    
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
                data_filter = data_int[-N:].reshape(-1,1).reshape(1,N,1)

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
        
    
def load_models():
    import tensorflow as tf
    '''
    Funcion que carga los modelos de maximo, media y regresion.
    '''
    tf.compat.v1.disable_eager_execution()
    models = ["mode_reg","model_max","model_mean"]
    M = []
    for i in models:
        try:
            M.append(tf.keras.models.load_model(f'models/{i}.h5'))
        except:
            logging.error('AI_error_5: modelo'+ str(i) +'no encontrado') 
        
    return M
    

#def real(inp,N = 200,n= 40):
#    real = inp[7][-N:-N+n].reshape(-1)
#    REAL = [real[0:int(n/2)],array([max(real)]),array([mean(real),mean(real) + std(real)])]  
#    return REAL
    

def predictions(inp,conv=3.6,alarm = 5, sam=15,N = 200,n= 40,ret = 1):
    
    
    
    '''
    Funcion que realiza la prediccion de valores medios, maximos y a lo largo del tiempo de los datos de input.
    
    Attributes
    ----------
    
    data : list
        datos provenientes de la funcion pipeline
    
    conv : float
        factor de conversion de unidades de los valores medidos por los sensores
    
    arlarm : float
        valor a partir del cual se activa la alarma
    
    real : list
        datos usados para realizar la validacion
    
    '''
    
    #cargado de los modelos    
    M = load_models()
    
    if inp is None or len(M) != 3:
        logging.error('AI_error_6: input de prediccion indecuado')
        
        if len(M) != 3: logging.error('AI_error_7: modelos no cargados')
        if inp is None: logging.error('AI_error_8: input no encontrado')
        
        logging.error('AI_fatal_error:imposible hacer prediccion')
        
        
        # return "error"
    
    else:
        
        
        #se ordenan los inputs
        pred_data,completeness,data_time,out_len,val_data,header = inp[0],inp[1],inp[2],inp[3],inp[4],inp[8]

        # data_time = inp[1][-1] - 15s* 400
        
        #predicciones
        Pre = [m.predict(pred_data)[0]*conv for m in M]#Regresion/Maximo/Media
        #Pre = []
        #for m in M:
        #   Pre.append(m.predict(pred_data)[0]*conv)



        n = len(Pre[0])
        #intervalo de confianza
        ic = [Pre[2][0], Pre[2][1] - Pre[2][0]]
        #alarma
        al = 0
        if Pre[2][0] > alarm:
            al = 1

        real = inp[7][-N:-N+n].reshape(-1)
        REAL = [real[0:n],array([max(real)]),array([mean(real),mean(real) + std(real)])]  
        
        #validacion
        if val_data is not None and real is not None:
            Val = [m.predict(val_data)[0]*conv for m in M]#Regresion/Maximo/Media
            
            
            #calculo del error de validacion en caso de que se cumpla que existan los datos de validacion
            #y que estos esten formateados de forma adecuada

            if (real is not None):
                
                
                error,val = [mean_absolute_percentage_error(Val[v].reshape(-1),REAL[v]*conv) for v in range(len(Val))],"yes"#Regresion/Maximo/Media
                # error,val = [[Val[v].reshape(-1),real[v]*conv] for v in range(len(Val))],"yes"#Regresion/Maximo/Media
                
                
                
                
            else:
            
                if real is None: logging.warning('AI_warning_1: no hay datos para realizar la validacion')
                if (len(concatenate(real)) != 24): logging.warning('AI_warning_2: datos de validacion sin formato adecuado: (20,1,2)')
                error,val  = [0,0,0],"no"
        else:
            logging.warning('AI_warning_3: datos a validar no encontrados')
            error,val = [0,0,0],"no"

        #formateo de las variables para el output
        values = (pred_data*conv).reshape(-1)
        now = datetime.datetime.now().replace(microsecond=0)
        
        d =  datetime.datetime.strptime(data_time[-1],'%Y-%m-%d %H:%M:%S+00:00')
        # d =  now
        
        time =  list(reversed([d - datetime.timedelta(seconds=sam*x) for x in range(2*N)]))
        time_ = [d + datetime.timedelta(seconds=sam) + datetime.timedelta(seconds=sam*x) for x in range(n)]
        tarray = array(time).astype(str).tolist()
        tarray_ =array(time_).astype(str).tolist()

        
        
        
        #guardo en un diccionario de la informacion
#         out = { 
#             "update_time":str(now),
#             "input":
#                 {"values":str(values.tolist()),
#                  "time": str(tarray),
#                  # "time_delta": str(dstar - d0),
#                  "mean":str(mean(values).tolist()),
#                  "max":str(max(values).tolist()),
#                  "varianza":str(var(values)),
#                  "QoI":
#                      {"Completeness":str(completeness),
#                       "Outliers":str(out_len)}

#                 },

#                "output":
#                    {"ConfInt": str(ic), 
#                    "MaxPred":str(Pre[1][0]),
#                     "Regression":str(Pre[0].tolist()),
#                    "time_reg": str(tarray_),
#                     "time_delta": str(time[-1] - time[0]),
#                     "alarm_value":str(alarm),
#                    "alarm": str(al),
#                    "validation":{
#                        "check": str(val),
#                         "regression":str(error[0]),
#                         "mean":str(error[1]),
#                         "maximum":str(error[2]),
#                         #"real":str(real),
#                         #"val":str(Val)}
#                    }}

#               }

        
        out = {
            "header": header,
            "update_time":str(now),
            "inputs":{
                # "input_0":str(list(inp[5])),
                "prepro":str(list(inp[0].reshape(-1))),
                "date_input":str(list(inp[2])),
            },
            
            "params":{
                "smooth":str(inp[6]),
                "conv":str(conv),
                "samp":str(sam),
                "alarm":str(alarm),
            },
            
            "QoI":{"Completeness":str(completeness),
                      "Outliers":str(out_len)},
            "output":{
                "ConfInt": str([ic[0], 1.96*ic[1]/(n**(1/2))]),
                "Max": str(Pre[1][0]),
                "Regression":str(Pre[0].tolist()),
                "time_reg":str(tarray_),
                "alarm_value": str(al),
                "validation":{
                    "check":str(val),
                    "MAPEreg":str(error[0]),
                    "MAPEmax":str(error[1]),
                    "MAPEmean":str(error[2])
                }
                
            }
            
        }
        return out
            
    

    
    
