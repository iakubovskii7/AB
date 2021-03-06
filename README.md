# Classic, Bayesian AB tests and multiarmed bandits for proba.ai

## Описание репозитория:
- *knowledge-base* содержит файлы с описанием алгоритмов, с которыми определились на данный момент
- *Description experiment results* содержит файлы с описанием результатов экспериментов
- В директории *Description experiment results* в папках 
Experiment1, Experiment2 и т.д. содержатся файлики с результатами
- bayes_classic_descr - описание тестирования классических Байесовских методов
- thompson_bayes_descr - описание тестирования алгоритма Томпсона и Байесовских методов
с остановкой
- methodology_description - описание методологии тестирования
- *notebooks* - различные эксперименты по алгоритмам и методам
- *src* - .py файлы с реализацией методов
- AB_tests_theory_lection - отличная лекция по классическим АБ-тестам
- main.py - файл для запуска экспериментов


## Основные виды алгоритмов, которые описываются в этом репозитории.

1. ***Классические АБ-тесты с поправкой на множественное тестирование.***
- Реализуются на любые метрики
- Определяется необходимое число юзеров в группы
- Для конверсии используется z-test для пропорций, для ARPU - тест Стьюдента
в случае нормальности выборок (или хотя бы нормальности средних). Универсальный 
метод проверки - бутстрапированные доверительные интервалы для разницы средних.
- Если доверительный интервал для разницы включает в себя 0, то статистических различий
нет.
- Вывод делаем 1 раз (нельзя подглядывать)
- Если вариантов больше двух, то используем поправку FDR (она описана в Description algorithms/AB_classic_algorithms.ipynb)

2. ***Классический байесовский алгоритм***
- Реализуется на конверсию (на данный момент)
- Собираем выборку из пользователей (примерно в 1.5 - 2 раза меньше, чем для обычного теста)
- Вероятность конверсии каждого варианта априорно задается с параметрами (1,1)
- После окончания эксперимента пересчитываем распределения вероятности конверсий и 
считаем вероятность превосходства
- Если вероятность превосходства выше заданного порога (тут клиент вправе его выбрать сам),
то определяем победителя

3. ***Байесовский тест с остановкой.***
- Не определяем число юзеров заранее
- Распределяем юзеров 50 на 50
- Используем байесовскую статистику пересчета вероятностей
- Можем остановить в любой момент при достижении определенных пороговых значений
выбранных критериев
- Примеры критерия: вероятность превосходства в каком-то варианте достигла 95%

4. ***Алгоритм Томпсона***
- Не определяем число юзеров заранее
- Распределяем юзеров в ходе эксперимента пропорционально вероятностям превосходства,
которые определяются аналогично алгоритмам 2 и 3.
- Останавливаем эксперимент аналогично алгоритму 3

## Как сопоставляются алгоритмы между собой (по убыванию показателя)

- Надежность выводов (мы четко знаем, в каком % случаев мы совершаем ошибки): 
1, 2, 3, 4
- Интерпретируемость выводов (что понятно клиенту): 4/3/2, 1
- Гибкость тестов: 3, 4, 2, 1
- Сложность реализации: 4, 3, 2, 1
- Скорость реализации (самый быстрый): 1, 2, 3/4


