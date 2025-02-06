{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "4a8f2108",
   "metadata": {},
   "source": [
    "# Тестовое задание для Maxbitsolution.\n",
    "### На позицию Machine learning engineer."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0028628b",
   "metadata": {},
   "source": [
    "### Цель:\n",
    "- построить DL модель для классификации состояния дерева (Good/Fair/Poor) по данным из NY street tree 2015."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f5f4c15a",
   "metadata": {},
   "source": [
    "Данные находятся в каталоге - data:\n",
    "- 2015-street-tree-census-tree-data.csv - исходный датасет.\n",
    "- train.csv - очищенные данные для тренировки.\n",
    "- test.csv - очищеные данные для проверки модели.\n",
    "\n",
    "Ноутбуки размещены в каталоге - notebooks:\n",
    "- eda_ny_tree.ipynb - exploratory data analysis и очистка данных.\n",
    "- baseline.ipynb - трансформация данных и модель градиентного бустинга из пакета sklearn.\n",
    "- tabnn.ipynb - DL модель на Pytorch.\n",
    "\n",
    "В папке models - сохраненные данный модели:\n",
    "- tabularnn_model.pth - модель Pytorch.\n",
    "- scaler.pkl и encoder.pkl - данный для предпроцессинга."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ce1971fe",
   "metadata": {},
   "source": [
    "Файл app.py представляет собой реализацию API на FastApi."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "35181dd5",
   "metadata": {},
   "source": [
    "### Запуск."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cc2daf2d",
   "metadata": {},
   "source": [
    "Для запуска используется unicorn:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "430982f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "uvicorn app:app --reload"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f0a506db",
   "metadata": {},
   "source": [
    "Можно протестировать API используя curl или Postman"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4cc996ad",
   "metadata": {},
   "outputs": [],
   "source": [
    "curl -X POST \"http://127.0.0.1:8000/predict\" \\\n",
    "-H \"Content-Type: application/json\" \\\n",
    "-d '{\"root_stone\": \"Yes\", \"root_grade\": \"No\", \"root_other\": \"No\", \"trunk_wire\": \"No\", \"trnk_ligth\": \"No\", \"trnk_other\": \"No\", \"brch_light\": \"Yes\", \"brch_shoe\": \"No\", \"brch_other\": \"No\", \"curb_loc\": \"OnCurb\", \"sidewalk\": \"Damage\", \"spc_common\":  \"honeylocust\", \"nta\": \"BK26\", \"tree_dbh\": 15}\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8231fc0d",
   "metadata": {},
   "source": [
    "Ответ:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ff25e6db",
   "metadata": {},
   "outputs": [],
   "source": [
    "{\n",
    "  \"prediction\": 'Good'\n",
    "}"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3b63c673",
   "metadata": {},
   "source": [
    "### Модель входных данных: \n",
    "Класс InputData определяет структуру входных данных с использованием BaseModel от Pydantic.\n",
    "### Предварительная обработка: \n",
    "Входные данные предварительно обрабатываются (Encoding, Scaling) перед передачей в модель.\n",
    "### Прогноз: \n",
    "Модель делает прогноз, и результат возвращается в виде ответа JSON. \n",
    "### Обработка ошибок: \n",
    "Если возникает ошибка (например, неверный ввод), возвращается ошибка HTTP 400 с подробной информацией."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a00f3e4c",
   "metadata": {},
   "source": [
    "## Модель TabularNN"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "be885833",
   "metadata": {},
   "source": [
    "#### Embedding \n",
    "Категориальные признаки передаются через слои внедрения для преобразования их в плотные векторы. \n",
    "#### Конкатенация\n",
    "Встроенные категориальные признаки и числовые признаки объединяются в один входной тензор.\n",
    "#### Линейные слои\n",
    "Объединенные объекты передаются через полностью связанные слои для составления прогнозов.\n",
    "#### Обучение\n",
    "Так как выходных классов 3, используется для обучения nn.CrossEntropyLoss"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5699937f",
   "metadata": {},
   "source": [
    "Такая архитектура выбрана потому, что позволяет использовать и категориальные и числовые данные, за которыми идет простой перцептрон."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ec95dcf0",
   "metadata": {},
   "source": [
    "### Выводы\n",
    "- Модель достигает точности 0.8106 или 81%.\n",
    "- Имеется возможность дальнейшего исследования модели, для роста точности."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
