# Aplicación de viento


Desarrollo del modulo de velocidad de viento, predicción de valores de velocidad de viento para mantener la seguridad de los módulos solares. Este sistema admite una serie de datos históricos de velocidad de viento y con el objetivo de entrenar una red neuronal, de forma que para obtener valores fururos de velocidad de viento, hará falta nada más que unos pocos valores en el pasado.

Descripción de los archivos:

* config.py: en este archivo se dan los parámetros necesarios para la ejecución de la aplicación, estos son:
  * FRANJA_HORARIA: franja horaria donde se encuentra la planta sobre la que se desea trabajar.
  * SAMPLEO: tiempo entre dato y dato.
  * DATOS_TRAIN: número de datos necesarios para el entrenamiento.
  * N_OUT: formato de los datos de salida.
  * SAVGOL: factor de suavizado.
  * N: datos del pasado.
  * n: datos predichos.
  * ALARM: valor a partir del cual salta la alarma de peligro.
  * CONV:  información sobre las unidades en las que se da la velocidad de viento.
  * EPOCHS: epocas del entrenamiento.
  * BATCH_SIZE: tamaño del lote.

* model.py: este archivo guarda funciones que definen los modelos de Deep Learning que se entrenarán con los datos históricos. Hay 2 funciones en su interior: model y Fmodel. La primera define el modelo y la segunda lo entrena.

* train.py: una vez definidos los modelos se procede al entrenamiento de los modelos, este script formatea los datos de de entrenamiento y los preprara para incluirlos para ajustar el modelo, posteriormente el modelo es entrenado.

* prepro.py: en este script se encuentran una serie de funciones:
  * interp: interpreta el json de input.
  * mean_squared_error/mean_absolute_percentage_error: definición de los errores de actuación del modelo.
  * savgol: factor de suavizado de los datos.
  * mad_method: característica no implementada.
  * pipeline: prepara los datos para la predicción formateándolos.

* prediction.py: script donde se realiza predicción de velocidad de viento. Las funciones con las que cuenta son las siguientes:
  * load_models: se cargan los modelos de regresión, máximo y media.
  * predictions: esta función es la más compleja porque realiza todo el proceso de formateo, predicción, cálculo de errores y exportación de los datos.

* test.py: 

    load_historic: carga datos históricos de un json.
    load_instance: carga un json de datos.
    isnumeric: comprueba si un dato tiene un valor numérico.
    isstring: comprueba si un dato es un string.
    isdate: comprueba si un dato es una fecha.
    error_pred: calcula el error de las últimas predicciones y lo compara con un valor.
    test_train: compreba si los valores para el entrenamiento son adecuados.
    test_format: comprueba si el formato de los datos es adecuado.
    test_nan: comprueba si hay una cantidad excesiva de NaN en los datos.
    test_prep: comprueba si el resultado del preprocesamiento es adecuado.
    test_predict: comprueba si el resultado de las predicciones es adecuado.
    test_error: compara errores con un valor umbral.




Ejemplo de input:

    {
      "header":{
        "empresa":"empresa",
        "id": "id_empresa",
        "planta":"1234"
        },
      "update_time": "2022-04-04 12:38:53.881619",
      "data":{
        "datetime": "['2021-06-06 00:00:00+00:00', '2021-06-06 00:00:13+00:00', '2021-06-06 00:00:28+00:00',...
        "values": "[0.74, 0.74, 0.71, 0.76, 0.82, 0.79,...
      }
    }


Ejemplo de output:

    {
      "header": {
        "empresa":"empresa",
        "id": "id_empresa",
        "planta":"1234"
        },
      "update_time": "2022-04-04 12:38:53.881619",
      "inputs":{
          "prepro":"[0.74, 0.74, 0.71, 0.76, 0.82, 0.79,...
          "date_input":"['2021-06-06 00:00:00+00:00', '2021-06-06 00:00:13+00:00',...
      },

      "params":{
          "smooth":31,
          "conv":3.6,
          "samp":15,
          "alarm":56,
      },

      "QoI":{"Completeness":1,
                "Outliers":0,
      "output":{
          "ConfInt": "[10,1]",
          "Max": "30",
          "Regression":"[0.74, 0.74, 0.71, 0.76, 0.82, 0.79,...
          "time_reg":"['2021-06-06 00:00:00+00:00', '2021-06-06 00:00:13+00:00',...
          "alarm_value": 1,
          "validation":{
              "check":"yes",
              "MAPEreg":0.1,
              "MAPEmax":0.1,
              "MAPEmean":0.1
          }

      }

    } 





