import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import pandas as pd
import skfuzzy.control as ctrl
import os

# Configuración de visualización
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

def cargar_datos_agricolas(filepath):
    """Carga y procesa datos agrícolas específicos"""
    try:
        df = pd.read_csv(filepath)
        
        print("\n🔍 Análisis inicial de datos:")
        print("Columnas disponibles:", df.columns.tolist())
        print(f"Total de registros: {len(df)}")
        print("\nPrimeras filas del dataset:")
        print(df.head())
        
        # Procesamiento según el dataset detectado
        if 'temperature' in df.columns:
            # Para Crop Recommendation Dataset
            temp = df['temperature'].mean()
            humedad = df['humidity'].mean()
            print("\n📊 Datos detectados: Crop Recommendation Dataset")
        elif 'Soil Moisture' in df.columns:
            # Para Agricultural Soil Data
            temp = df['Temperature'].mean()
            humedad = df['Soil Moisture'].mean()
            print("\n📊 Datos detectados: Agricultural Soil Data")
        else:
            # Datos por defecto si no se reconocen las columnas
            print("\n⚠️ Formato no reconocido, usando valores por defecto")
            temp = 25.0
            humedad = 60.0
        
        print(f"\n🌡️ Temperatura promedio: {temp:.2f}°C")
        print(f"💧 Humedad promedio: {humedad:.2f}%")
        
        return temp, humedad
    
    except Exception as e:
        print(f"\n❌ Error al cargar datos: {str(e)}")
        return 25.0, 60.0  # Valores por defecto

def configurar_sistema_difuso():
    """Configura el sistema de lógica difusa"""
    # Universos de discurso
    humedad_range = np.arange(0, 101, 1)
    temp_range = np.arange(0, 51, 1)
    etapa_range = np.arange(0, 101, 1)
    output_range = np.arange(10, 41, 1)
    
    # Variables difusas
    humedad = ctrl.Antecedent(humedad_range, 'humedad')
    temperatura = ctrl.Antecedent(temp_range, 'temperatura')
    etapa = ctrl.Antecedent(etapa_range, 'etapa')
    optima = ctrl.Consequent(output_range, 'optima')
    
    # Funciones de membresía (usando trapezoidales para suaves transiciones)
    humedad['baja'] = fuzz.trapmf(humedad_range, [0, 0, 30, 50])
    humedad['media'] = fuzz.trapmf(humedad_range, [30, 50, 70, 90])
    humedad['alta'] = fuzz.trapmf(humedad_range, [70, 90, 100, 100])
    
    temperatura['fria'] = fuzz.trapmf(temp_range, [0, 0, 10, 20])
    temperatura['optima'] = fuzz.trapmf(temp_range, [10, 20, 30, 40])
    temperatura['caliente'] = fuzz.trapmf(temp_range, [30, 40, 50, 50])
    
    etapa['inicial'] = fuzz.trapmf(etapa_range, [0, 0, 30, 60])
    etapa['final'] = fuzz.trapmf(etapa_range, [40, 70, 100, 100])
    
    optima['enfriar'] = fuzz.trapmf(output_range, [10, 10, 15, 25])
    optima['mantener'] = fuzz.trapmf(output_range, [15, 25, 30, 35])
    optima['calentar'] = fuzz.trapmf(output_range, [30, 35, 40, 40])
    
    # Visualización de las funciones de membresía
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(8, 12))
    
    humedad.view(ax=ax0)
    ax0.set_title('Humedad del suelo (%)')
    
    temperatura.view(ax=ax1)
    ax1.set_title('Temperatura ambiente (°C)')
    
    etapa.view(ax=ax2)
    ax2.set_title('Etapa de crecimiento')
    
    optima.view(ax=ax3)
    ax3.set_title('Temperatura óptima recomendada')
    
    plt.tight_layout()
    plt.show()
    
    return humedad, temperatura, etapa, optima

def crear_reglas(humedad, temperatura, etapa, optima):
    """Crea el conjunto de reglas difusas"""
    rules = [
        # Reglas basadas en humedad y temperatura
        ctrl.Rule(humedad['baja'] & temperatura['fria'], optima['calentar']),
        ctrl.Rule(humedad['media'] & temperatura['optima'], optima['mantener']),
        ctrl.Rule(humedad['alta'] & temperatura['caliente'], optima['enfriar']),
        
        # Reglas basadas en etapa de crecimiento
        ctrl.Rule(etapa['inicial'], optima['mantener']),
        ctrl.Rule(etapa['final'] & temperatura['caliente'], optima['enfriar']),
        
        # Regla por defecto
        ctrl.Rule(temperatura['optima'], optima['mantener'])
    ]
    return rules

def ejecutar_simulacion(cultivo, filepath, etapa_crecimiento):
    """Ejecuta la simulación completa"""
    print(f"\n🌾 Simulación para cultivo: {cultivo.upper()}")
    
    # Cargar datos agrícolas
    temp_ambiente, humedad_suelo = cargar_datos_agricolas(filepath)
    
    # Validación de entradas
    etapa_crecimiento = np.clip(etapa_crecimiento, 0, 100)
    temp_ambiente = np.clip(temp_ambiente, 0, 50)
    humedad_suelo = np.clip(humedad_suelo, 0, 100)
    
    print(f"\n📊 Entradas del sistema:")
    print(f"- Humedad del suelo: {humedad_suelo:.2f}%")
    print(f"- Temperatura ambiente: {temp_ambiente:.2f}°C")
    print(f"- Etapa de crecimiento: {etapa_crecimiento:.2f}%")
    
    # Configurar sistema difuso
    humedad, temperatura, etapa, optima = configurar_sistema_difuso()
    
    # Crear reglas
    rules = crear_reglas(humedad, temperatura, etapa, optima)
    
    # Crear y ejecutar simulador
    sistema = ctrl.ControlSystem(rules)
    simulador = ctrl.ControlSystemSimulation(sistema)
    
    # Asignar entradas
    simulador.input['humedad'] = humedad_suelo
    simulador.input['temperatura'] = temp_ambiente
    simulador.input['etapa'] = etapa_crecimiento
    
    # Computar resultados
    simulador.compute()
    
    # Obtener y mostrar resultados
    temp_optima = simulador.output['optima']
    print(f"\n✅ Recomendación para {cultivo}:")
    print(f"Temperatura óptima objetivo: {temp_optima:.2f}°C")
    
    # Visualización de resultados
    optima.view(sim=simulador)
    plt.title(f'Recomendación de temperatura para {cultivo}')
    plt.show()

# Ejecución principal
if __name__ == "__main__":
    # Configuración de rutas (ajustar según necesidad)
    dataset_path = 'Crop_recommendation.csv'  # Reemplazar con tu archivo
    
    # Verificar existencia del archivo
    if not os.path.exists(dataset_path):
        print(f"\n❌ Error: No se encontró el archivo {dataset_path}")
        print("Por favor descarga uno de los datasets recomendados:")
        print("1. Crop Recommendation Dataset: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
        print("2. Agricultural Soil Data: https://www.kaggle.com/datasets/brsdincer/agricultural-soil")
    else:
        # Ejecutar simulación
        cultivo = "maíz"  # Puede cambiarse según el cultivo a analizar
        etapa = 75  # Valor entre 0-100 (0=inicio, 100=cosecha)
        
        ejecutar_simulacion(cultivo, dataset_path, etapa)