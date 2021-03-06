# Описание экспериментов

Здесь будет описание результатов для 2-х похожих алгоритмов: 
- Байесовский алгоритм с остановкой со сплитом 50 на 50 
- Алгоритм Томпсона с остановкой со сплитом в разных пропорциях 

Единственное отличие этих алгоритмов - пропорция сплита. В остальном он работает так:
1. В качестве априорного распределения для вероятности конверсий используются бета-распределения с параметрами (1, 1)
2. В ходе эксперимента пересчитываем вероятности победить для каждого варианта, исходя из числа успешных конверсий
3. Останавливаем алгоритм и определяем победителя в зависимости от определенных критериев:

- вероятность победить выше определенного значения
- суммарное число юзеров перевалило за какое-то количество (ниже, чем при классических тестах)
- ожидаемые потери не выходят за определенное число на протяжении какого-то количества времени
- какая-то связь между "credible intervals"

## Эксперимент 1

Алгоритм: 
- перераспределяем в пропорциях вероятности победить
- добавляем условие 2^l < k

***Результат***

1. Выяснил, что алгоритм не совсем верно распределял сплит по юзерам +
в статье имелось ввиду, что условие 2^l < k должно быть внутри батча 

## Эксперимент 2
- перераспределяем в пропорциях (для Байеса просто 50 на 50)
- критерий остановки - вероятность превосходства ИЛИ число юзеров__ по одной руке в 2 раза больше нужного
- если тест заканчивается на числе юзеров, то выбираем вариант с большей вероятностью
превосходства

***Результат***
1. Для больших батчей мощность тестов лучше (очень странный результат)
2. Довольно высокая мощность для вероятности превосходства в 90%
3. Мощность выше для случаев Байесовских тестов с остановкой
4. Мощность падает до 65-70% в случае Томпсона 

## Эксперимент 3
- перераспределяем в пропорциях (для Байеса просто 50 на 50)
- критерий остановки - суммарное число юзеров по 2 рукам достигает необходимого числа юзеров
- победителя выбираем по максимальной вероятности превосходства

***Результат***
1. Для больших батчей мощность тестов лучше (очень странный результат)
2. Мощность тестов падает по сравнению с экспериментов 2
3. Самая низкая мощность - для размера батчей в 1% от суммарного числа юзеров (35%)

## Эксперимент 4
- перераспределяем в пропорциях (для Байеса просто 50 на 50) 
- критерий остановки - только пороговая вероятность превосходства

***Результат***
1. Эксперимент может просто не заканчиваться, если ориентироваться только
на вероятность превосходства. При некоторых датасетах вероятность превосходства
при большом числе юзеров не доходит даже до 90%. 
