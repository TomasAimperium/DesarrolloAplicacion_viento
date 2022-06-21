import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import datetime
import json
from pipeline import *
import warnings
from IPython.display import clear_output
import config as c
import glob


f = c.FRANJA_HORARIA#"+00:00"
s = c.SAMPLEO#15
N_data = c.DATOS_TRAIN#100000
n_out = c.N_OUT#[2,20]
sav = c.SAVGOL
N = c.N
n = c.n
alarm = c.ALARM
conv = c.CONV


def load_historic():
	f = open('data/historico.json')
	js = json.load(f)
	D = pd.DataFrame([json.loads(js['data']['values']),ast.literal_eval(js['data']['datetime'])]).T
	return D.iloc[:N_data,:]
	

def load_instance(i):
	files = glob.glob("data/jsons/*")
	f = open(files[i])
	js = json.load(f)
	return js



def isnumeric(s):
	try:
	    float(s)
	    return True
	except ValueError:
	    return False

def isstring(s):
	try:
	    str(s)
	    return True
	except ValueError:
	    return False

def isdate(date_text):
    try:
        datetime.datetime.strptime(date_text, "%Y-%m-%d %H:%M:%S" + f)
        return True
    except ValueError:
        return False


def error_pred():
	val = []
	for i in range(5):
		DATA = load_instance(i)
		inp = pipeline(DATA,sav = sav,N = N,n= n)
		out = predictions(inp,conv=conv,alarm = alarm, sam=s, N = N,n= n,ret = 1) 
		print(i+1,float(out['output']['validation']['MAPEmax']))
		e = (float(out['output']['validation']['MAPEmax']) + 
			float(out['output']['validation']['MAPEreg']) +
			float(out['output']['validation']['MAPEmean']))/3


	val.append(e)
	return np.mean(val)


def test_train():

	D = load_historic()
	assert len(D) == N_data


	assert np.sum([isnumeric(s) for s in D.values[:,0]]) / len(D.values[:,0]) == 1
	assert np.isnan(list(D.values[:,0])).sum() / len(D.values[:,0]) <= 0.85
	try:
		fechas = [datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S"+f) for i in D.iloc[:,1]]
		assert np.mean(np.diff(fechas)).total_seconds() <= (s+5)
	except:
		print("Tiempo de sampleo irregular")


def test_format():
	assert data.shape == (2*N,2)
	assert np.sum([isnumeric(s) for s in data[:,0]]) / len(data[:,0]) == 1
	assert np.sum([isstring(s) for s in data[:,1]]) / len(data[:,1]) == 1 
	assert np.sum([isdate(s) for s in data[:,1]]) / len(data[:,1]) == 1 

def test_nan():
	assert np.isnan(list(data[:,0])).sum() / len(data[:,0]) <= 0.85


#def test_sample():
#	fechas = [datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S"+f) for i in data[:,0]]
#	assert np.mean(np.diff(fechas)).total_seconds() < (s+5)



def test_prep():
	DATA = load_instance(i)
	inp = pipeline(DATA,sav = sav,N = N,n= n)
	assert type(inp) == list
	assert len(inp) == 9
	assert inp[0].shape == (1,N,1)
	assert np.sum([isnumeric(s) for s in inp[0].reshape(-1)]) / len(inp[0].reshape(-1)) == 1
	assert type(inp[1]) == float
	assert len(inp[2])== 2*N
	assert np.sum([isstring(s) for s in inp[2]]) / len(inp[2]) == 1 
	assert np.sum([isdate(s) for s in inp[2]]) / len(inp[2]) == 1 	
	assert type(inp[3]) == int
	assert inp[4].shape == (1,N,1)
	assert np.sum([isnumeric(s) for s in inp[4][0]]) / len(inp[4][0]) == 1

def test_predict():

	DATA = load_instance(i)
	inp = pipeline(DATA,sav = sav,N = N,n= n)
	out = predictions(inp,conv=3.6,alarm = alarm, sam=s, N = N,n= n,ret = 1) 
	assert type(out) == dict
	assert len(json.loads(out['output']['ConfInt'])) == 2
	assert type(json.loads(out['output']['Max'])) == float
	assert len(json.loads(out['output']['Regression'])) == n/2
	assert np.isnan(json.loads(out['output']['ConfInt'])).sum() == 0
	assert np.isnan(json.loads(out['output']['Max'])).sum() == 0
	assert np.isnan(json.loads(out['output']['Regression'])).sum() == 0


def test_error():
	#en otro momento esto sera una simple consulta a la base de datos
	er = error_pred()
	assert er <= 15




if __name__ == "__main__":

	D = load_historic()	
	nn = 10
	ind = np.random.randint(0,1000,nn)

######################################################
	print("#############################")
	print("Test de nan")
	print("...")	
	for i in range(nn):
		
		data = D.iloc[ind[i]:(ind[i]+2*N),:].values
		test_nan()
	print("La completitud es adecuada")
	print("")
	
######################################################
	print("#############################")
	print("Test de formato")
	print("...")
	for i in range(nn):
		
		data = D.iloc[ind[i]:(ind[i]+2*N),:].values
		test_format()
	print("El formato es adecuado")
	print("")


######################################################

	print("#############################")
	print("Test de preprocesamiento")
	print("...")
	for i in range(nn):
		data = load_instance(ind[i])
		test_prep()
	print("El preprocesamiento es adecuado")
	print("#############################")
	print("")



######################################################
	print("#############################")
	print("Test de predicciones")
	print("...")
	for i in range(nn):		
		print(i+1,"/",nn)
		
		data = load_instance(ind[i])
		test_predict()
		

	print("Las predicciones son adecuadas")
######################################################
	print("#############################")
	print("")
	test_train()

	try:
		test_train()
		print("#############################")
		print("Modelo entrenable")
		print("#############################")
		print("")

	except:

		print("#############################")
		print("Modelo no entrenable por datos insuficientes")
		print("#############################")
		print("")

######################################################
	print("#############################")
	print("Check de entranemiento del modelo")
	print("...")
	
	try:
		test_error()
		print("El modelo esta dando buenos resultados")
	except:
		print("El modelo debe ser entrenado")

	print("#############################")








