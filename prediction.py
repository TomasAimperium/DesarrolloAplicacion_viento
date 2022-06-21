import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



from numpy import sum,array,median,abs,nan,ndarray,concatenate,mean,max,var,std,isnan
import datetime
import logging
from tensorflow.compat.v1 import disable_eager_execution
from tensorflow.keras.models import load_model
from prepro import *

    
def load_models():
    
    #import tensorflow as tf


    '''
    Funcion que carga los modelos de maximo, media y regresion.
    '''
    disable_eager_execution()
    models = ["mode_reg","model_max","model_mean"]
    M = []
    for i in models:
        try:
            M.append(load_model(f'models/{i}.h5'))
        except:
            logging.error('AI_error_5: modelo'+ str(i) +'no encontrado') 
        
    return M
    
    

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
        Pre = [m.predict(pred_data[:,:N,:])[0] for m in M]#Regresion/Maximo/Media
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
            Val = [m.predict(val_data)[0] for m in M]#Regresion/Maximo/Media
            
            
            #calculo del error de validacion en caso de que se cumpla que existan los datos de validacion
            #y que estos esten formateados de forma adecuada

            if (real is not None):
                
                
                error,val = [mean_absolute_percentage_error(Val[v].reshape(-1),REAL[v]) for v in range(len(Val))],"yes"#Regresion/Maximo/Media
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

        
        
        out = {
            "header": header,
            "update_time":str(d),
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
            
    

    
    
