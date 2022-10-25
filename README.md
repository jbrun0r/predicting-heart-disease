# Machine Learning - Predição de Doenças Cardíacas.

        - Análise Exploratória de Dados;
        - dataset 'heart.csv';
        - Correlações, Remoção de outilers, duplicates, [...];
        - Aplicação de Diferentes Algoritmos de Aprendizagem de Máquina;
        - Observação dos Resultados.

# Prelúdio


```python
# Manipulação e análise de dados.
import pandas as pd

# Processamento de grandes, multi-dimensionais arranjos e matrizes;
# Coleção de funções matemáticas de alto nível.
import numpy as np

# Criação de gráficos e visualizações de dados em geral.
import matplotlib.pyplot as plt
%matplotlib inline

# Visualização de dados Python baseada em matplotlib;
# Interface de alto nível para desenhar gráficos estatísticos atraentes e informativos.
import seaborn as sns

# Algoritmos fundamentais para computação científica em Python.
from scipy import stats

```


```python
# Dividir matrizes ou matrizes em subconjuntos aleatórios de treinamento e teste.
from sklearn.model_selection import train_test_split

# Pontuação de classificação de precisão;
# Relatório de texto mostrando as principais métricas de classificação;
# Calcular a matriz de confusão para avaliar a precisão de uma classificação.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Algoritmos de Machine Learning a serem implementados.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB

# Padronização das features removendo a média e dimensionando para a variação da unidade.
from sklearn.preprocessing import StandardScaler
```

### Heart Disease Dataset:


```python
# Consumindo o Dataset.
ds = pd.read_csv('/home/jbrun0r/Documentos/scripts/datasets/heart.csv')
ds_copy = pd.read_csv('/home/jbrun0r/Documentos/scripts/datasets/heart.csv')
```

##### age: Idade do paciente
##### sex: Sexo do paciente
    0: F;
    1: M.
##### cp : tipo de dor no peito no peito
    1: angina típica;
    2: angina atípica;
    3: dor não anginosa;
    4: assintomático.
##### trtbps : pressão arterial em repouso (em mm Hg)
##### chol : colestoral em mg/dl obtido via sensor de IMC(Colesterol sérico em mg/dl.)
##### fbs : (glicemia em jejum > 120 mg/dl) (Açúcar no sangue em jejum > 120 mg/dl = 0 | < 120mg/dl = 1.)
    1 = verdadeiro;
    0 = falso.
##### rest_ecg : Resultados eletrocardiográficos em repouso
    0: normal;
    1: com anormalidade da onda ST-T (inversões da onda T e/ou elevação ou depressão do segmento ST > 0,05 mV);
    2: mostrando provável ou definitiva hipertrofia ventricular esquerda pelos critérios de Estes.
##### thalach : Frequência cardíaca máxima atingida
    0 = menos chance de ataque cardíaco
    1 = mais chance de ataque cardíaco
##### exng: Angina induzida pelo exercício
    1 = sim
    0 = não
##### Oldpeak: Depressão do segmento ST induzida pelo exercício em relação ao repouso
##### slp: Inclinação do segmento ST de pico do exercício.
##### caa: número de vasos principais
    0-3
##### thal: Talassemia: 
    1 = normal;
    2 = problema corrigido; 
    3 = problema reversível.
##### output:
    0 = não possui doença cardíaca;
    1 = possui doença cardíaca.
##### Importante: Angina é o nome dado para a dor no peito causada pela diminuição do fluxo de sangue no coração, o que é chamado de isquemia. Ela não é uma doença, mas está relacionada a outras condições que provocam obstrução nas artérias coronárias, responsáveis por levar sangue ao coração.


```python
ds.head()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trtbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalachh</th>
      <th>exng</th>
      <th>oldpeak</th>
      <th>slp</th>
      <th>caa</th>
      <th>thall</th>
      <th>output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>1</td>
      <td>3</td>
      <td>145</td>
      <td>233</td>
      <td>1</td>
      <td>0</td>
      <td>150</td>
      <td>0</td>
      <td>2.3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>1</td>
      <td>2</td>
      <td>130</td>
      <td>250</td>
      <td>0</td>
      <td>1</td>
      <td>187</td>
      <td>0</td>
      <td>3.5</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>0</td>
      <td>1</td>
      <td>130</td>
      <td>204</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
      <td>0</td>
      <td>1.4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
      <td>1</td>
      <td>1</td>
      <td>120</td>
      <td>236</td>
      <td>0</td>
      <td>1</td>
      <td>178</td>
      <td>0</td>
      <td>0.8</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>120</td>
      <td>354</td>
      <td>0</td>
      <td>1</td>
      <td>163</td>
      <td>1</td>
      <td>0.6</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



# Pré-Processamento


```python
ds.isnull().sum()
#Abaixo podemos verificar se existem registros nulos em nossa base de dados, 
# e para a nossa sorte nenhum foi encontrado.
```




    age         0
    sex         0
    cp          0
    trtbps      0
    chol        0
    fbs         0
    restecg     0
    thalachh    0
    exng        0
    oldpeak     0
    slp         0
    caa         0
    thall       0
    output      0
    dtype: int64




```python
ds.min()
# Podemos ver que também não possuímos valores negativos nos dados.
```




    age          29.0
    sex           0.0
    cp            0.0
    trtbps       94.0
    chol        126.0
    fbs           0.0
    restecg       0.0
    thalachh     71.0
    exng          0.0
    oldpeak       0.0
    slp           0.0
    caa           0.0
    thall         0.0
    output        0.0
    dtype: float64




```python
ds_copy.duplicated().sum()
# Podemos ver que há 'uma' linha duplicada e tiraremos para não causar overfit
```




    1




```python
#Removendo a duplicação
ds = ds.drop_duplicates()
ds_copy = ds_copy.drop_duplicates()
```


```python
ds.describe().transpose()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>302.0</td>
      <td>54.420530</td>
      <td>9.047970</td>
      <td>29.0</td>
      <td>48.00</td>
      <td>55.5</td>
      <td>61.00</td>
      <td>77.0</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>302.0</td>
      <td>0.682119</td>
      <td>0.466426</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>cp</th>
      <td>302.0</td>
      <td>0.963576</td>
      <td>1.032044</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>trtbps</th>
      <td>302.0</td>
      <td>131.602649</td>
      <td>17.563394</td>
      <td>94.0</td>
      <td>120.00</td>
      <td>130.0</td>
      <td>140.00</td>
      <td>200.0</td>
    </tr>
    <tr>
      <th>chol</th>
      <td>302.0</td>
      <td>246.500000</td>
      <td>51.753489</td>
      <td>126.0</td>
      <td>211.00</td>
      <td>240.5</td>
      <td>274.75</td>
      <td>564.0</td>
    </tr>
    <tr>
      <th>fbs</th>
      <td>302.0</td>
      <td>0.149007</td>
      <td>0.356686</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>restecg</th>
      <td>302.0</td>
      <td>0.526490</td>
      <td>0.526027</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>thalachh</th>
      <td>302.0</td>
      <td>149.569536</td>
      <td>22.903527</td>
      <td>71.0</td>
      <td>133.25</td>
      <td>152.5</td>
      <td>166.00</td>
      <td>202.0</td>
    </tr>
    <tr>
      <th>exng</th>
      <td>302.0</td>
      <td>0.327815</td>
      <td>0.470196</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>oldpeak</th>
      <td>302.0</td>
      <td>1.043046</td>
      <td>1.161452</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.8</td>
      <td>1.60</td>
      <td>6.2</td>
    </tr>
    <tr>
      <th>slp</th>
      <td>302.0</td>
      <td>1.397351</td>
      <td>0.616274</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>2.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>caa</th>
      <td>302.0</td>
      <td>0.718543</td>
      <td>1.006748</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>thall</th>
      <td>302.0</td>
      <td>2.314570</td>
      <td>0.613026</td>
      <td>0.0</td>
      <td>2.00</td>
      <td>2.0</td>
      <td>3.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>output</th>
      <td>302.0</td>
      <td>0.543046</td>
      <td>0.498970</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



###### count: Total de registros existentes.
###### mean: valor da média.
###### std: desvio padrão.
###### min e max: valores mínimos e máximos.
###### 25%: ou primeiro quartil, que corresponde aos primeiros 25% dos valores de nossos registros.
###### 50%: ou mediana, que corresponde aos primeiros 50% dos valores de nossos registros, essa é a nossa medida central dos dados. Metade dos nossos valores são menores que a mediana e a outra metade é maior a este valor. A mediana é uma opção muito interessante quando comparado a média, pois ela não sofre com valores discrepantes.
###### 75%: ou terceiro quartil, que corresponde aos primeiros 75% dos valores de nossos registros.



```python
ds.corr()
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>cp</th>
      <th>trtbps</th>
      <th>chol</th>
      <th>fbs</th>
      <th>restecg</th>
      <th>thalachh</th>
      <th>exng</th>
      <th>oldpeak</th>
      <th>slp</th>
      <th>caa</th>
      <th>thall</th>
      <th>output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>1.000000</td>
      <td>-0.094962</td>
      <td>-0.063107</td>
      <td>0.283121</td>
      <td>0.207216</td>
      <td>0.119492</td>
      <td>-0.111590</td>
      <td>-0.395235</td>
      <td>0.093216</td>
      <td>0.206040</td>
      <td>-0.164124</td>
      <td>0.302261</td>
      <td>0.065317</td>
      <td>-0.221476</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>-0.094962</td>
      <td>1.000000</td>
      <td>-0.051740</td>
      <td>-0.057647</td>
      <td>-0.195571</td>
      <td>0.046022</td>
      <td>-0.060351</td>
      <td>-0.046439</td>
      <td>0.143460</td>
      <td>0.098322</td>
      <td>-0.032990</td>
      <td>0.113060</td>
      <td>0.211452</td>
      <td>-0.283609</td>
    </tr>
    <tr>
      <th>cp</th>
      <td>-0.063107</td>
      <td>-0.051740</td>
      <td>1.000000</td>
      <td>0.046486</td>
      <td>-0.072682</td>
      <td>0.096018</td>
      <td>0.041561</td>
      <td>0.293367</td>
      <td>-0.392937</td>
      <td>-0.146692</td>
      <td>0.116854</td>
      <td>-0.195356</td>
      <td>-0.160370</td>
      <td>0.432080</td>
    </tr>
    <tr>
      <th>trtbps</th>
      <td>0.283121</td>
      <td>-0.057647</td>
      <td>0.046486</td>
      <td>1.000000</td>
      <td>0.125256</td>
      <td>0.178125</td>
      <td>-0.115367</td>
      <td>-0.048023</td>
      <td>0.068526</td>
      <td>0.194600</td>
      <td>-0.122873</td>
      <td>0.099248</td>
      <td>0.062870</td>
      <td>-0.146269</td>
    </tr>
    <tr>
      <th>chol</th>
      <td>0.207216</td>
      <td>-0.195571</td>
      <td>-0.072682</td>
      <td>0.125256</td>
      <td>1.000000</td>
      <td>0.011428</td>
      <td>-0.147602</td>
      <td>-0.005308</td>
      <td>0.064099</td>
      <td>0.050086</td>
      <td>0.000417</td>
      <td>0.086878</td>
      <td>0.096810</td>
      <td>-0.081437</td>
    </tr>
    <tr>
      <th>fbs</th>
      <td>0.119492</td>
      <td>0.046022</td>
      <td>0.096018</td>
      <td>0.178125</td>
      <td>0.011428</td>
      <td>1.000000</td>
      <td>-0.083081</td>
      <td>-0.007169</td>
      <td>0.024729</td>
      <td>0.004514</td>
      <td>-0.058654</td>
      <td>0.144935</td>
      <td>-0.032752</td>
      <td>-0.026826</td>
    </tr>
    <tr>
      <th>restecg</th>
      <td>-0.111590</td>
      <td>-0.060351</td>
      <td>0.041561</td>
      <td>-0.115367</td>
      <td>-0.147602</td>
      <td>-0.083081</td>
      <td>1.000000</td>
      <td>0.041210</td>
      <td>-0.068807</td>
      <td>-0.056251</td>
      <td>0.090402</td>
      <td>-0.083112</td>
      <td>-0.010473</td>
      <td>0.134874</td>
    </tr>
    <tr>
      <th>thalachh</th>
      <td>-0.395235</td>
      <td>-0.046439</td>
      <td>0.293367</td>
      <td>-0.048023</td>
      <td>-0.005308</td>
      <td>-0.007169</td>
      <td>0.041210</td>
      <td>1.000000</td>
      <td>-0.377411</td>
      <td>-0.342201</td>
      <td>0.384754</td>
      <td>-0.228311</td>
      <td>-0.094910</td>
      <td>0.419955</td>
    </tr>
    <tr>
      <th>exng</th>
      <td>0.093216</td>
      <td>0.143460</td>
      <td>-0.392937</td>
      <td>0.068526</td>
      <td>0.064099</td>
      <td>0.024729</td>
      <td>-0.068807</td>
      <td>-0.377411</td>
      <td>1.000000</td>
      <td>0.286766</td>
      <td>-0.256106</td>
      <td>0.125377</td>
      <td>0.205826</td>
      <td>-0.435601</td>
    </tr>
    <tr>
      <th>oldpeak</th>
      <td>0.206040</td>
      <td>0.098322</td>
      <td>-0.146692</td>
      <td>0.194600</td>
      <td>0.050086</td>
      <td>0.004514</td>
      <td>-0.056251</td>
      <td>-0.342201</td>
      <td>0.286766</td>
      <td>1.000000</td>
      <td>-0.576314</td>
      <td>0.236560</td>
      <td>0.209090</td>
      <td>-0.429146</td>
    </tr>
    <tr>
      <th>slp</th>
      <td>-0.164124</td>
      <td>-0.032990</td>
      <td>0.116854</td>
      <td>-0.122873</td>
      <td>0.000417</td>
      <td>-0.058654</td>
      <td>0.090402</td>
      <td>0.384754</td>
      <td>-0.256106</td>
      <td>-0.576314</td>
      <td>1.000000</td>
      <td>-0.092236</td>
      <td>-0.103314</td>
      <td>0.343940</td>
    </tr>
    <tr>
      <th>caa</th>
      <td>0.302261</td>
      <td>0.113060</td>
      <td>-0.195356</td>
      <td>0.099248</td>
      <td>0.086878</td>
      <td>0.144935</td>
      <td>-0.083112</td>
      <td>-0.228311</td>
      <td>0.125377</td>
      <td>0.236560</td>
      <td>-0.092236</td>
      <td>1.000000</td>
      <td>0.160085</td>
      <td>-0.408992</td>
    </tr>
    <tr>
      <th>thall</th>
      <td>0.065317</td>
      <td>0.211452</td>
      <td>-0.160370</td>
      <td>0.062870</td>
      <td>0.096810</td>
      <td>-0.032752</td>
      <td>-0.010473</td>
      <td>-0.094910</td>
      <td>0.205826</td>
      <td>0.209090</td>
      <td>-0.103314</td>
      <td>0.160085</td>
      <td>1.000000</td>
      <td>-0.343101</td>
    </tr>
    <tr>
      <th>output</th>
      <td>-0.221476</td>
      <td>-0.283609</td>
      <td>0.432080</td>
      <td>-0.146269</td>
      <td>-0.081437</td>
      <td>-0.026826</td>
      <td>0.134874</td>
      <td>0.419955</td>
      <td>-0.435601</td>
      <td>-0.429146</td>
      <td>0.343940</td>
      <td>-0.408992</td>
      <td>-0.343101</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Observando a correlação das features;
# "Correlação não implica em Causalidade.".
plt.figure(figsize=(16,8))
sns.heatmap(ds.corr(), annot = True, cmap='Blues')
```








    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_16_1.png)
    



```python
categoricas = ['sex', 'cp', 'fbs','restecg','exng', 'slp','caa', 'thall']
numericas = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
```


```python
for c in ds_copy.columns:
    plt.figure(figsize=(12,4))
    plt.title(f'Coluna avaliada: {c}')
    if c in categoricas:
        sns.countplot(x = ds_copy[c], hue= ds.output)
        plt.show()

        n = ds_copy[c].loc[ds_copy.output == 0].value_counts()
        y = ds_copy[c].loc[ds_copy.output == 1].value_counts()

        print('     output 0: ')
        for i in range(len(n)):
            print(f'            {c} '+str(i)+ ' = '+str(n[i])+' -> {:.4}'.format((n[i]/len(ds_copy[c].loc[ds_copy.output==0]))*100)+'%')
        print('')
        print('     output 1: ')
        for i in range(len(y)):
            print(f'            {c} '+str(i)+' = '+str(y[i])+' -> {:.4}'.format((y[i]/len(ds_copy[c].loc[ds_copy.output==1]))*100)+'%')
        print('')
        print('     GERAL: ')
        for i in range(len(n)):
            print(f'            {c} '+str(i)+' e output 0 = '+str(n[i])+' -> {:.4}'.format((n[i]/len(ds_copy[c]))*100)+'%')
            print(f'            {c} '+str(i)+' e output 1 = '+str(y[i])+' -> {:.4}'.format((y[i]/len(ds_copy[c]))*100)+'%')
        print('')

    if c in numericas:
        sns.histplot(ds_copy[c], kde=True)
```


    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_18_0.png)
    



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_18_1.png)
    


         output 0: 
                sex 0 = 24 -> 17.39%
                sex 1 = 114 -> 82.61%
    
         output 1: 
                sex 0 = 72 -> 43.9%
                sex 1 = 92 -> 56.1%
    
         GERAL: 
                sex 0 e output 0 = 24 -> 7.947%
                sex 0 e output 1 = 72 -> 23.84%
                sex 1 e output 0 = 114 -> 37.75%
                sex 1 e output 1 = 92 -> 30.46%
    



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_18_3.png)
    


         output 0: 
                cp 0 = 104 -> 75.36%
                cp 1 = 9 -> 6.522%
                cp 2 = 18 -> 13.04%
                cp 3 = 7 -> 5.072%
    
         output 1: 
                cp 0 = 39 -> 23.78%
                cp 1 = 41 -> 25.0%
                cp 2 = 68 -> 41.46%
                cp 3 = 16 -> 9.756%
    
         GERAL: 
                cp 0 e output 0 = 104 -> 34.44%
                cp 0 e output 1 = 39 -> 12.91%
                cp 1 e output 0 = 9 -> 2.98%
                cp 1 e output 1 = 41 -> 13.58%
                cp 2 e output 0 = 18 -> 5.96%
                cp 2 e output 1 = 68 -> 22.52%
                cp 3 e output 0 = 7 -> 2.318%
                cp 3 e output 1 = 16 -> 5.298%
    



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_18_5.png)
    



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_18_6.png)
    



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_18_7.png)
    


         output 0: 
                fbs 0 = 116 -> 84.06%
                fbs 1 = 22 -> 15.94%
    
         output 1: 
                fbs 0 = 141 -> 85.98%
                fbs 1 = 23 -> 14.02%
    
         GERAL: 
                fbs 0 e output 0 = 116 -> 38.41%
                fbs 0 e output 1 = 141 -> 46.69%
                fbs 1 e output 0 = 22 -> 7.285%
                fbs 1 e output 1 = 23 -> 7.616%
    



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_18_9.png)
    


         output 0: 
                restecg 0 = 79 -> 57.25%
                restecg 1 = 56 -> 40.58%
                restecg 2 = 3 -> 2.174%
    
         output 1: 
                restecg 0 = 68 -> 41.46%
                restecg 1 = 95 -> 57.93%
                restecg 2 = 1 -> 0.6098%
    
         GERAL: 
                restecg 0 e output 0 = 79 -> 26.16%
                restecg 0 e output 1 = 68 -> 22.52%
                restecg 1 e output 0 = 56 -> 18.54%
                restecg 1 e output 1 = 95 -> 31.46%
                restecg 2 e output 0 = 3 -> 0.9934%
                restecg 2 e output 1 = 1 -> 0.3311%
    



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_18_11.png)
    



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_18_12.png)
    


         output 0: 
                exng 0 = 62 -> 44.93%
                exng 1 = 76 -> 55.07%
    
         output 1: 
                exng 0 = 141 -> 85.98%
                exng 1 = 23 -> 14.02%
    
         GERAL: 
                exng 0 e output 0 = 62 -> 20.53%
                exng 0 e output 1 = 141 -> 46.69%
                exng 1 e output 0 = 76 -> 25.17%
                exng 1 e output 1 = 23 -> 7.616%
    



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_18_14.png)
    



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_18_15.png)
    


         output 0: 
                slp 0 = 12 -> 8.696%
                slp 1 = 91 -> 65.94%
                slp 2 = 35 -> 25.36%
    
         output 1: 
                slp 0 = 9 -> 5.488%
                slp 1 = 49 -> 29.88%
                slp 2 = 106 -> 64.63%
    
         GERAL: 
                slp 0 e output 0 = 12 -> 3.974%
                slp 0 e output 1 = 9 -> 2.98%
                slp 1 e output 0 = 91 -> 30.13%
                slp 1 e output 1 = 49 -> 16.23%
                slp 2 e output 0 = 35 -> 11.59%
                slp 2 e output 1 = 106 -> 35.1%
    



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_18_17.png)
    


         output 0: 
                caa 0 = 45 -> 32.61%
                caa 1 = 44 -> 31.88%
                caa 2 = 31 -> 22.46%
                caa 3 = 17 -> 12.32%
                caa 4 = 1 -> 0.7246%
    
         output 1: 
                caa 0 = 130 -> 79.27%
                caa 1 = 21 -> 12.8%
                caa 2 = 7 -> 4.268%
                caa 3 = 3 -> 1.829%
                caa 4 = 3 -> 1.829%
    
         GERAL: 
                caa 0 e output 0 = 45 -> 14.9%
                caa 0 e output 1 = 130 -> 43.05%
                caa 1 e output 0 = 44 -> 14.57%
                caa 1 e output 1 = 21 -> 6.954%
                caa 2 e output 0 = 31 -> 10.26%
                caa 2 e output 1 = 7 -> 2.318%
                caa 3 e output 0 = 17 -> 5.629%
                caa 3 e output 1 = 3 -> 0.9934%
                caa 4 e output 0 = 1 -> 0.3311%
                caa 4 e output 1 = 3 -> 0.9934%
    



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_18_19.png)
    


         output 0: 
                thall 0 = 1 -> 0.7246%
                thall 1 = 12 -> 8.696%
                thall 2 = 36 -> 26.09%
                thall 3 = 89 -> 64.49%
    
         output 1: 
                thall 0 = 1 -> 0.6098%
                thall 1 = 6 -> 3.659%
                thall 2 = 129 -> 78.66%
                thall 3 = 28 -> 17.07%
    
         GERAL: 
                thall 0 e output 0 = 1 -> 0.3311%
                thall 0 e output 1 = 1 -> 0.3311%
                thall 1 e output 0 = 12 -> 3.974%
                thall 1 e output 1 = 6 -> 1.987%
                thall 2 e output 0 = 36 -> 11.92%
                thall 2 e output 1 = 129 -> 42.72%
                thall 3 e output 0 = 89 -> 29.47%
                thall 3 e output 1 = 28 -> 9.272%
    



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_18_21.png)
    


## Análise:

##### caa: existem entradas incorrentas, segundo o link da kaggle para este dataset, caa deveria se comportar de 0 a 3, porém observamos que há um tipo de entrada incorreta, trataremos isto:


```python
ds.caa.loc[ds.caa == 4] = 0
```

##### tha: existem entradas incorrentas, segundo o link da kaggle para este dataset, thall deveria se comportar de 1 a 3, porém observamos que há um tipo de entrada incorreta, trataremos isto:


```python
ds.thall.loc[ds.thall == 0] = 2
```

##### Podemos observar outliers nas numericas com os histogramas, para evidenciar confira o boxplot:


```python
plt.figure(figsize=(18,8))
sns.boxplot(data = ds_copy)
```




    <AxesSubplot:>




    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_25_1.png)
    


### Tratamento dos outliers

##### trtbps: 


```python
plt.figure(figsize=(12,4))
sns.boxplot(ds_copy.trtbps)
plt.show

# Tratamento dos outliers
#ds.trtbps.quantile(0.97)
ds.trtbps.loc[ds.trtbps > 170] = 170

# Plotando novo boxplot após o tratamento de outliers
plt.figure(figsize=(12,4))
sns.boxplot(ds.trtbps)
plt.show()

ds.trtbps.describe().transpose()
```

    /usr/local/lib/python3.10/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_28_1.png)
    



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_28_2.png)
    





    count    302.000000
    mean     131.258278
    std       16.605232
    min       94.000000
    25%      120.000000
    50%      130.000000
    75%      140.000000
    max      170.000000
    Name: trtbps, dtype: float64



##### chol: 


```python
# Há presença de outliers
plt.figure(figsize=(12,4))
sns.boxplot(ds_copy.chol)
plt.show()

# Tratando os outliers
#ds_copy.chol.quantile(0.98338871)
ds.chol.loc[ds.chol > 360] = 360

# Plotagem com outliers tratados
plt.figure(figsize=(12,4))
sns.boxplot(ds.chol)
plt.show()

ds.chol.describe().transpose()
```

    /usr/local/lib/python3.10/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_30_1.png)
    


    /usr/local/lib/python3.10/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_30_3.png)
    





    count    302.000000
    mean     245.205298
    std       47.049535
    min      126.000000
    25%      211.000000
    50%      240.500000
    75%      274.750000
    max      360.000000
    Name: chol, dtype: float64



##### thalachh


```python
plt.figure(figsize=(12, 4))
sns.boxplot(ds_copy.thalachh)

# Tratamento do outlier
# Limite Inferior do boxplot = Primeiro Quartil (– 1,5 * (Terceiro Quartil – Primeiro Quartil))
limite_inferior = ( ds.thalachh.quantile(0.25)) -1.5 * (ds.thalachh.quantile(0.75)-ds.thalachh.quantile(0.25))
# o limite_inferior é igual a 84.75
# quase ds_copy.thalachh.quantile(0.00254)

ds.thalachh.loc[ds.thalachh < limite_inferior] = limite_inferior
# Outlier tratado

# Plot novo boxplot
plt.figure(figsize=(12, 4))
sns.boxplot(ds.thalachh)
plt.show()

ds.thalachh.describe().transpose()
```

    /usr/local/lib/python3.10/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    /usr/local/lib/python3.10/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_32_1.png)
    



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_32_2.png)
    





    count    302.000000
    mean     149.612997
    std       22.765983
    min       84.125000
    25%      133.250000
    50%      152.500000
    75%      166.000000
    max      202.000000
    Name: thalachh, dtype: float64



### Outliers e Entradas-Incorretas devidamente tratadas


```python
plt.figure(figsize=(18,8))
sns.boxplot(data = ds)
plt.show()
```


    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_34_0.png)
    


##### numericas = 
    ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
##### categoricas = 
    ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall']


```python
ds1 = pd.get_dummies(ds, columns = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall'])
ds1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>trtbps</th>
      <th>chol</th>
      <th>thalachh</th>
      <th>oldpeak</th>
      <th>output</th>
      <th>sex_0</th>
      <th>sex_1</th>
      <th>cp_0</th>
      <th>cp_1</th>
      <th>...</th>
      <th>slp_0</th>
      <th>slp_1</th>
      <th>slp_2</th>
      <th>caa_0</th>
      <th>caa_1</th>
      <th>caa_2</th>
      <th>caa_3</th>
      <th>thall_1</th>
      <th>thall_2</th>
      <th>thall_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>145</td>
      <td>233</td>
      <td>150.0</td>
      <td>2.3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>130</td>
      <td>250</td>
      <td>187.0</td>
      <td>3.5</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41</td>
      <td>130</td>
      <td>204</td>
      <td>172.0</td>
      <td>1.4</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56</td>
      <td>120</td>
      <td>236</td>
      <td>178.0</td>
      <td>0.8</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>57</td>
      <td>120</td>
      <td>354</td>
      <td>163.0</td>
      <td>0.6</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>298</th>
      <td>57</td>
      <td>140</td>
      <td>241</td>
      <td>123.0</td>
      <td>0.2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>299</th>
      <td>45</td>
      <td>110</td>
      <td>264</td>
      <td>132.0</td>
      <td>1.2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>300</th>
      <td>68</td>
      <td>144</td>
      <td>193</td>
      <td>141.0</td>
      <td>3.4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>301</th>
      <td>57</td>
      <td>130</td>
      <td>131</td>
      <td>115.0</td>
      <td>1.2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>302</th>
      <td>57</td>
      <td>130</td>
      <td>236</td>
      <td>174.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>302 rows × 29 columns</p>
</div>



# Aprendizagem

Function learn:

        Dividir o dataset em subconjuntos de treinamento e teste;
        test size = 33% e treino size = 67%;
        Normatizar as Features numéricas com StandardScaler;
        Invocar Algoritmos de Machine Learning;
        Gerar Relatório mostrando as principais métricas de classificação;
        Calcular a matriz de confusão para avaliar a precisão de uma classificação.



```python
def learn(dataset, algoritmo, opt = 2):
    X = dataset.drop('output', axis=1)
    y = dataset.output

    # train_test_split

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

    # Padronização das features numéricas com StandardScaler;
    # Os modelos Não Baseados em Árvores de Decisão se beneficiam mais deste tipo de padronização.
    scaler = StandardScaler()

    columns_scaler = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']

    X_train[columns_scaler] = scaler.fit_transform(X_train[columns_scaler])
    X_test[columns_scaler] = scaler.transform(X_test[columns_scaler])
    

    if opt == 0:
        ml = algoritmo(max_iter = 1000)
    elif opt == 1:
        ml = algoritmo(n_estimators = 1000)
    elif opt == 2:
        ml = algoritmo()

    # training
    ml.fit(X_train, y_train)
    print('Acurácia:')
    score_train = ml.score(X_train, y_train)
    print('     Treino = {:.4}'.format(score_train*100)+'%')

    score_test = ml.score(X_test, y_test)
    print('     Teste = {:.4}'.format(score_test*100)+'%')

    # predict

    y_predict = ml.predict(X_test)
    print('\nClassification:\n',classification_report(y_test, y_predict))
    print('Confusion Matrix:')
    confusion = confusion_matrix(y_test, y_predict)
    sns.heatmap(confusion, annot=True, cmap='Blues')
    
    return score_train, score_test
```

### Invocando Algoritmos de Machine Learning:

     KNeighborsClassifier:


```python
kn_train, kn_test = learn(ds1, KNeighborsClassifier)
```

    Acurácia:
         Treino = 89.6%
         Teste = 84.0%
    
    Classification:
                   precision    recall  f1-score   support
    
               0       0.80      0.84      0.82        43
               1       0.87      0.84      0.86        57
    
        accuracy                           0.84       100
       macro avg       0.84      0.84      0.84       100
    weighted avg       0.84      0.84      0.84       100
    
    Confusion Matrix:



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_42_1.png)
    


    LogisticRegression: 


```python
log_train, log_test = learn(ds1, LogisticRegression,0)
```

    Acurácia:
         Treino = 86.63%
         Teste = 87.0%
    
    Classification:
                   precision    recall  f1-score   support
    
               0       0.84      0.86      0.85        43
               1       0.89      0.88      0.88        57
    
        accuracy                           0.87       100
       macro avg       0.87      0.87      0.87       100
    weighted avg       0.87      0.87      0.87       100
    
    Confusion Matrix:



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_44_1.png)
    


    DecisionTreeClassifier:


```python
tree_train, tree_test = learn(ds1, DecisionTreeClassifier)
```

    Acurácia:
         Treino = 100.0%
         Teste = 75.0%
    
    Classification:
                   precision    recall  f1-score   support
    
               0       0.69      0.77      0.73        43
               1       0.81      0.74      0.77        57
    
        accuracy                           0.75       100
       macro avg       0.75      0.75      0.75       100
    weighted avg       0.76      0.75      0.75       100
    
    Confusion Matrix:



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_46_1.png)
    


    AdaBoostClassifier:


```python
ada_train, ada_test = learn(ds1, AdaBoostClassifier)
```

    Acurácia:
         Treino = 95.54%
         Teste = 83.0%
    
    Classification:
                   precision    recall  f1-score   support
    
               0       0.78      0.84      0.81        43
               1       0.87      0.82      0.85        57
    
        accuracy                           0.83       100
       macro avg       0.83      0.83      0.83       100
    weighted avg       0.83      0.83      0.83       100
    
    Confusion Matrix:



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_48_1.png)
    


    RandomForestClassifier:


```python
rand_train, rand_test = learn(ds1, RandomForestClassifier)
```

    Acurácia:
         Treino = 100.0%
         Teste = 85.0%
    
    Classification:
                   precision    recall  f1-score   support
    
               0       0.80      0.86      0.83        43
               1       0.89      0.84      0.86        57
    
        accuracy                           0.85       100
       macro avg       0.85      0.85      0.85       100
    weighted avg       0.85      0.85      0.85       100
    
    Confusion Matrix:



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_50_1.png)
    


    MLPClassifier:


```python
mlp_train, mlp_test = learn(ds1, MLPClassifier)
```

    /usr/local/lib/python3.10/dist-packages/sklearn/neural_network/_multilayer_perceptron.py:702: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
      warnings.warn(


    Acurácia:
         Treino = 88.12%
         Teste = 89.0%
    
    Classification:
                   precision    recall  f1-score   support
    
               0       0.88      0.86      0.87        43
               1       0.90      0.91      0.90        57
    
        accuracy                           0.89       100
       macro avg       0.89      0.89      0.89       100
    weighted avg       0.89      0.89      0.89       100
    
    Confusion Matrix:



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_52_2.png)
    


    BernoulliNB:


```python
bnb_train, bnb_test = learn(ds1, BernoulliNB)
```

    Acurácia:
         Treino = 86.14%
         Teste = 83.0%
    
    Classification:
                   precision    recall  f1-score   support
    
               0       0.80      0.81      0.80        43
               1       0.86      0.84      0.85        57
    
        accuracy                           0.83       100
       macro avg       0.83      0.83      0.83       100
    weighted avg       0.83      0.83      0.83       100
    
    Confusion Matrix:



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_54_1.png)
    


    GaussianNB:


```python
gnb_train, gnb_test = learn(ds1, GaussianNB)
```

    Acurácia:
         Treino = 79.7%
         Teste = 81.0%
    
    Classification:
                   precision    recall  f1-score   support
    
               0       0.85      0.67      0.75        43
               1       0.79      0.91      0.85        57
    
        accuracy                           0.81       100
       macro avg       0.82      0.79      0.80       100
    weighted avg       0.82      0.81      0.81       100
    
    Confusion Matrix:



    
![png](https://github.com/jbrun0r/predicting-heart-disease/blob/outputs/output_56_1.png)
    


### Criando um rank das Acurácias Test:


```python
dados = {'Models' :['LogisticRegression',
                    'DecisionTreeClassifier',
                    'KNeighborsClassifier',
                    'RandomForestClassifier',
                    'AdaBoostClassifier',
                    'MLPClassifier',
                    'GaussianNB',
                    'BernoulliNB'],
                    
'Accuracy Train' : [round(log_train*100,2),
                    round(tree_train*100,2),
                    round(kn_train*100,2),
                    round(rand_train*100,2),
                    round(ada_train*100,2),
                    round(mlp_train*100,2),
                    round(gnb_train*100,2),
                    round(bnb_train*100,2)],

'Accuracy Test' :  [round(log_test*100,2),
                    round(tree_test*100,2),
                    round(kn_test*100,2),
                    round(rand_test*100,2),
                    round(ada_test*100,2),
                    round(mlp_test*100,2),
                    round(gnb_test*100,2),
                    round(bnb_test*100,2)],}

rank = pd.DataFrame(dados)
rank.sort_values(by='Accuracy Test', ascending=False, inplace = True)
```

# Rank da Aprendizagem:


```python
rank
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Models</th>
      <th>Accuracy Train</th>
      <th>Accuracy Test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>MLPClassifier</td>
      <td>88.12</td>
      <td>89.0</td>
    </tr>
    <tr>
      <th>0</th>
      <td>LogisticRegression</td>
      <td>86.63</td>
      <td>87.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RandomForestClassifier</td>
      <td>100.00</td>
      <td>85.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KNeighborsClassifier</td>
      <td>89.60</td>
      <td>84.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AdaBoostClassifier</td>
      <td>95.54</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>BernoulliNB</td>
      <td>86.14</td>
      <td>83.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>GaussianNB</td>
      <td>79.70</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DecisionTreeClassifier</td>
      <td>100.00</td>
      <td>75.0</td>
    </tr>
  </tbody>
</table>
</div>





MIT License

Copyright (c) 2022 João Bruno

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
