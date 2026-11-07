import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Подборка вин', page_icon='🍷', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(ROOT / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(ROOT / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


def predict_batch(frame, model, columns, threshold):
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError('Не хватает столбцов: ' + ', '.join(missing))
    if frame.empty:
        raise ValueError('Файл не содержит строк.')
    if frame.columns.duplicated().any():
        raise ValueError('Названия столбцов должны быть уникальными.')
    features = frame.loc[:, columns].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError('Признаки должны быть числами без пропусков и бесконечностей.')
    if not features['is_red'].isin([0, 1]).all():
        raise ValueError('is_red должен быть равен 0 (белое) или 1 (красное).')
    result = frame.copy()
    result['probability'] = model.predict_proba(features)[:, 1]
    result['prediction'] = (result['probability'] >= threshold).astype(int)
    return result


model = load_model()
data = load_test()
features = list(model.feature_names_in_)
X = data[features]
y = data['target']
probability = test_proba(model, X)
config = json.loads((ROOT / 'config.json').read_text())
threshold = st.sidebar.slider('Порог отбора', 0.0, 1.0, float(config['threshold']),
                              step=0.001, key='threshold')
st.title('🍷 Подборка хороших вин')
st.caption('Хорошее вино: экспертная оценка не ниже 7 из 10.')
single, quality, batch, weights = st.tabs(
    ['Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

with single:
    values = {}
    with st.form('wine'):
        columns = st.columns(3)
        for i, name in enumerate(features):
            with columns[i % 3]:
                if name == 'is_red':
                    values[name] = int(st.selectbox('Тип вина', ['Белое', 'Красное']) == 'Красное')
                else:
                    values[name] = st.number_input(name, min_value=float(X[name].min()),
                        max_value=float(X[name].max()), value=float(X[name].median()), format='%.4f')
        st.form_submit_button('Оценить')
    p = float(model.predict_proba(pd.DataFrame([values], columns=features))[0, 1])
    st.metric('Вероятность хорошего вина', f'{p:.1%}')
    st.write('Рекомендуем в подборку' if p >= threshold else 'Не включаем в подборку')

with quality:
    prediction = probability >= threshold
    precision = metrics.precision_score(y, prediction, zero_division=0)
    recall = metrics.recall_score(y, prediction, zero_division=0)
    f1 = metrics.f1_score(y, prediction, zero_division=0)
    for column, label, value in zip(st.columns(3), ['Precision', 'Recall', 'F1'], [precision, recall, f1]):
        column.metric(label, f'{value:.3f}')
    st.write('При повышении порога подборка становится меньше: обычно растёт точность, '
             'но больше хороших вин пропускается. При снижении порога охват растёт, '
             'однако в подборку попадает больше посредственных вин.')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    metrics.ConfusionMatrixDisplay.from_predictions(y, prediction, ax=axes[0], colorbar=False)
    axes[0].set_title('Матрица ошибок')
    metrics.RocCurveDisplay.from_predictions(y, probability, ax=axes[1])
    tn, fp, fn, tp = metrics.confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    axes[1].scatter(fp/(fp+tn), recall, color='red', label=f't={threshold:.3f}')
    axes[1].legend()
    metrics.PrecisionRecallDisplay.from_predictions(y, probability, ax=axes[2])
    axes[2].axhline(y.mean(), color='gray', linestyle='--')
    axes[2].scatter(recall, precision, color='red', label=f't={threshold:.3f}')
    axes[2].legend()
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

with batch:
    st.write('Загрузите CSV с 12 признаками вина. Используется порог из боковой панели.')
    st.download_button('Скачать пример CSV', X.head(5).to_csv(index=False).encode('utf-8'),
                       file_name='example.csv', mime='text/csv')
    uploaded = st.file_uploader('CSV с признаками', type=['csv'])
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded)
            result = predict_batch(frame, model, features, threshold)
            st.dataframe(result)
            st.download_button('Скачать прогноз', result.to_csv(index=False).encode('utf-8'),
                               file_name='predictions.csv', mime='text/csv')
        except (ValueError, TypeError, pd.errors.ParserError, UnicodeError) as error:
            st.error(f'Не удалось обработать CSV: {error}')

with weights:
    coefficients = pd.Series(model[-1].coef_.ravel(), index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    coefficients.plot.barh(ax=ax, color=np.where(coefficients >= 0, 'seagreen', 'indianred'))
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Коэффициент после StandardScaler')
    st.pyplot(fig); plt.close(fig)
    st.write('Сильнее за хорошее вино: ' + ', '.join(coefficients.nlargest(3).index) + '.')
    st.write('Сильнее против: ' + ', '.join(coefficients.nsmallest(3).index) + '.')
    st.caption('Коэффициенты показывают связь при прочих фиксированных признаках, а не причинное влияние.')
