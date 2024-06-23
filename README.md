# Проект по One-hot encoding
Для начало установим все нужные библиотеки 
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
откроем таблцу:
```
taxiDB = pd.read_csv('taxi_dataset.csv')
```
![krasivo](https://i.postimg.cc/CKD6FW2k/2024-05-25-020410.png)

Описание колонок:


id - ID поездки 

vendor_id - ID компании, осуществляющей перевозку 

pickup_datetime - Таймкод начала поездки

dropoff_datetime - Таймкод конца поездки 
passenger_count - Количество пассажиров 

pickup_longitude - Долгота точки, в которой началась поездка 

pickup_latitude - Широта точки, в которой началась поездка 

dropoff_longitude - Долгота точки, в которой закончилась поездка 

dropoff_latitude - Широта точки, в которой закончилась поездка 

store_and_fwd_flag - Yes/No: Была ли информация сохранена в памяти транспортного средства из-за потери соединения с сервером 

Чтоб выполнить польностью задание нужно действовать по шагам.  Давайте их выполнять вместе :)
___
**1 шаг**

**Наша целевая переменная - длительность поездки.**

Зная тайм-коды времени начала и конца поездок, можем вычислить обозначенный таргет
Договоримся, что производим вычисления в секундах.
Советую обратить внимание на  <a href="https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html">данный способ</a> для перевода строки в datetime тип, с которым удобно работать при вычленении дней/часов...

И <a href="https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.total_seconds.html"> этот </a>для перевода разницы datetime объектов в секунды

Положи таргетную переменнул в столбик с названием **trip_duration**
```
import pandas as pd

taxiDB['pickup_datetime'] = pd.to_datetime(taxiDB['pickup_datetime'])
taxiDB['dropoff_datetime'] = pd.to_datetime(taxiDB['dropoff_datetime'])
taxiDB["trip_duration"] = taxiDB["dropoff_datetime"] - taxiDB["pickup_datetime"]
       
taxiDB.head()
```
![krasivo](https://i.postimg.cc/SNzmB3t6/2024-05-26-232121.png)

Я выполнил 1 шаг а именно перевел столбик **trip_duration** в секунды 
___
**2 шаг**

Предсказывая таргет для новых объектов в будущем, мы не будем заранее знать **dropoff_datetime**.

Удалим колонку из датасета.

```
taxiDB = taxiDB.drop(["dropoff_datetime"], axis=1)
taxiDB.head()
```
![krasivo](https://i.postimg.cc/RZ4VwGXw/2024-05-26-232518.png)
Удалив столбик **dropoff_datetime** я выполнил 2 шаг
___
**3 шаг**
 
 **В будущем будем строить модель. На каких признаках? Рассмотрим имеющиеся вещественные/бинарные и обсудим, какие простейшие признаки можно вытащить из остальных колонок.**

Во-первых, имеем бинарный признак vendor_id, принимающий значения {1, 2}. Переведем его во множество {0, 1}, так как это просто привычнее.

```
taxiDB['vendor_id'] = taxiDB['vendor_id'] - 1
taxiDB.head()
```
![krasivo](https://i.postimg.cc/x8GVCVrh/2024-05-26-232823.png)
И так я уже решил 3 шаг находя **бинарный признак** я сделал чтоб он принимал только {0,1}
___
**4 шаг**

Найдя еще один **бинарный признак** в данном датасете. Закодируйте и его тоже во множество {0, 1}.

```
taxiDB['store_and_fwd_flag'] = taxiDB['store_and_fwd_flag'] = 1
taxiDB["store_and_fwd_flag"].value_counts()
```
**Вывод**
```
store_and_fwd_flag
1    1458644
Name: count, dtype: int64
```
Находя еще один **бинарный признак** я решил уже 4 шаг и по **выводу** можно видеть что в столбце только одно значение **No** (позже это нам понадобиться )
_____

**5 шаг**

**Начнем переводить каждую долготу в некоторое относительно километровое выражение**

```
allLat  = list(taxiDB['pickup_latitude']) + list(taxiDB['dropoff_latitude'])

allLat 
```
**Вывод**

```
[40.76793670654297,
 40.738563537597656,
 40.763938903808594,
 40.719970703125,
 40.79320907592773,
 40.74219512939453,
 40.75783920288086,
 40.79777908325195,
 40.738399505615234,
 40.744338989257805,
 40.76383972167969,
 40.74943923950195,
 40.7566795349121,
 40.76794052124024,
 40.72722625732422,
 40.768592834472656,
 40.75556182861328,
 40.745803833007805,
 40.7130126953125,
 40.73819732666016,
 40.7424201965332,
 40.753360748291016,
 40.7588119506836,
 40.747173309326165,
 40.77713394165039,
...
 40.74602127075195,
 40.76163864135742,
 40.7879295349121,
 40.733497619628906,
 ...]
```
И так мы собрали список из всех широт (как точек старта, так и конца).

И мы уже выполнили 5 шаг!

___

**6 шаг**

Посчитаем медиану:

```
medianLat  = sorted(allLat)[int(len(allLat)/2)]

medianLat
```

**Вывод**

```
40.75431823730469
```

И так мы нашли медиану, выполнив 6 шаг

_____

**7 шаг**

Теперь, для из каждого значения широты вычтем медианное значение.


Результат переведем в километры.

```
latMultiplier  = 111.32

taxiDB['pickup_latitude']   = latMultiplier  * (taxiDB['pickup_latitude']   - medianLat)


taxiDB['dropoff_latitude']   = latMultiplier  * (taxiDB['dropoff_latitude']  - medianLat)


taxiDB['pickup_latitude'] 
```

**Вывод**

```
0          1.516008
1         -1.753813
2          1.070973
3         -3.823568
4          4.329328
             ...   
1458639   -0.979248
1458640   -0.772442
1458641    1.611979
1458642   -0.585171
1458643    3.053673
```

```
taxiDB['dropoff_latitude']
```

**Вывод**

```

0          1.256121
1         -2.578912
2         -4.923841
3         -5.298809
4          3.139453
             ...   
1458639   -1.575035
1458640    4.700899
1458641   -5.226193
1458642    0.310421
1458643    4.037168
```

7 шаг выполнен! 

**Мы почтим у цели**
______

**8 шаг**

Итого, для **latitude** колонок получили следующие выражения:

**На сколько примерно километров севернее или южнее (в зависимости от знака) точка находится относительно средней широты**

```
allLong = list(taxiDB['pickup_longitude']) + list(taxiDB['dropoff_longitude'])

medianLong  = sorted(allLong)[int(len(allLong)/2)]

longMultiplier = np.cos(medianLat*(np.pi/180.0)) * 111.32
longMultiplier 
```

**Вывод**

```
84.32665722947378
```

И так мы ответив на вопрос сможем подойти к **9 шагу**

_______

**9 шаг**

**Используя полученную медиану и множитель, на который стоит корректировать все долготы, получите корректные **longitude** признаки по аналогии.**

```
taxiDB['pickup_latitude'] = longMultiplier * (taxiDB['pickup_latitude'] - medianLong) 
taxiDB['dropoff_latitude'] =longMultiplier * (taxiDB['dropoff_latitude'] - medianLong) 
```

**Вывод**

```
0          6366.397685
1          6090.664596
2          6328.869337
3          5916.129132
4          6603.635569
              ...     
1458639    6155.981109
1458640    6173.420332
1458641    6374.490630
1458642    6189.212318
1458643    6496.063855
Name: pickup_latitude, Length: 1458644, dtype: float64 0          6344.482275
1          6021.086753
2          5823.346738
3          5791.726956
4          6503.297373
              ...     
1458639    6105.740392
1458640    6634.968875
1458641    5797.850379
1458642    6264.734537
1458643    6578.998639
Name: dropoff_latitude, Length: 1458644, dtype: float64

```

И так мы получили корректные **longitude** признаки и можем перейти к **10 шагу**

__________

**10 шаг**

Наконец, вычислим географическое расстояние **distance_km**:

```
from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371.0
    return c * r

haversine_1 = np.vectorize(haversine)

taxiDB["distanse_km"] = haversine_1(taxiDB['pickup_latitude'], taxiDB['pickup_longitude'], taxiDB['dropoff_latitude'], taxiDB['dropoff_longitude'])

taxiDB["distanse_km"]
```

**Вывод**

```
0           668.999831
1          2013.481824
2          3396.059838
3          3136.294008
4          2722.225632
              ...     
1458639    1495.460941
1458640    2744.298915
1458641    3377.258869
1458642    2163.971005
1458643    2342.279478
Name: distanse_km, Length: 1458644, dtype: float64
```
 
И так мы смогли вычислить географическоке расстаяние с помощью формулы **haversine** <a href="https://www.askpython.com/python/examples/calculate-gps-distance-using-haversine-formula">подробнее</a>
И мы уже на **11 шагу**
___________

**11 шаг**

Для начало выведим таблицу (то что у нас поулчилось :) )

```
taxiDB.head()
```

![krasivoo](https://i.postimg.cc/5N53B6CL/2024-06-23-234403.png)

теперь удалим старые признаки:

```
taxiDB = taxiDB.drop(['pickup_longitude', 'dropoff_longitude',
                      'pickup_latitude', 'dropoff_latitude'], axis=1)

```

![krasivooo](https://i.postimg.cc/2y05YQtn/2024-06-23-234641.png)

Таблицы в разы уменьшилось! Что значит очень хорошо :)

______________

**12 шаг**

В-третьих, обратим внимание на колонку **passenger_count**.

Какие значения она может принимать?

```
passenger_count_unique = taxiDB["passenger_count"].unique()
passenger_count_unique
```
**Вывод**

```
array([1, 6, 4, 2, 3, 5, 0, 7, 9, 8], dtype=int64)
```

Мы поняли какие значение она омжет принимтаь и переходим к **13 шагу**

____

**13 шаг**

Сейчас я заменю колонку **passenger_count** на колонку **category_encoded**

```
mean_passenger_count = taxiDB.groupby('passenger_count').size().mean()
passenger_count_dict = taxiDB.groupby('passenger_count').size().to_dict()
taxiDB['category_encoded'] = taxiDB['passenger_count'].map(passenger_count_dict)
```

И вот мы уже близки к таблицце состояющую из чиселок 

Переходим к заключающемся **шагу 14**
__________

**14 шаг**

И так остались 2 колонки не с числами а именно **id**, **pickup_datetime**. 

И так для начало м давайте дропнем колонку **passenger_count** потому что помните я вам говорил про то что везде значение **No** ну вот оно нам и пригодилось :) 

```
taxiDB.drop(columns=['passenger_count'], inplace=True)
```

Есть дропнули!

Столбик **id** можно использовать в качестве индекса нашей таблицы:

```
taxiDB = taxiDB.set_index('id')
```

Ну а **pickup_datetime** оставим :)

В итоге у нас получилось:

![krasivooo](https://i.postimg.cc/SNXwGLG3/2024-06-23-235745.png)

Я считаю это очень хороший результат! А посмотреть мой код ваы сможете в файлле под названием **HW1new_1_1.ipynb**



























