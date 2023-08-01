import bokeh.models as bm, bokeh.plotting as pl
from bokeh.io import output_notebook
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE


def plot_importance(chain, colnames, classes, label=None, top_n=20, ):
    if label:
        label_id = np.where(classes == label)[0][0]
        est = chain.estimators_[label_id]
        coefs = est.coef_.flatten()
        if label_id == 0:
            pass
        else:
            coefs = coefs[:-label_id]
        
        name = label
    else:
      all_coefs = np.array([np.abs(est.coef_.flatten()[:-i]) if i != 0 else np.abs(est.coef_.flatten()) for i, est in enumerate(chain.estimators_)])
      #[print(est.coef_.flatten()[:-i].shape) for i, est in enumerate(chain.estimators_)]
      coefs = np.mean(all_coefs, axis=0)
      name = 'all'
    print(label_id)
    importance = pd.DataFrame({
        'abs_weight': coefs,
        'feature': colnames
    })
    imp20 = importance.sort_values(by='abs_weight', ascending=False)[:top_n]
    ax = sns.barplot(y='feature', x='abs_weight', data=imp20, orient='h');
    ax.set(xlabel=f'Absolut weights for {name} class')
    

def draw_vectors(x, y, radius=10, alpha=0.25, color='blue',
                 width=600, height=400, show=True, **kwargs):
    """ draws an interactive plot for data points with auxilirary info on hover """
    output_notebook()

    if isinstance(color, str):
        color = [color] * len(x)
    data_source = bm.ColumnDataSource({ 'x' : x, 'y' : y, 'color': color, **kwargs })

    fig = pl.figure(active_scroll='wheel_zoom', width=width, height=height)
    fig.scatter('x', 'y', size=radius, color='color', alpha=alpha, source=data_source)

    fig.add_tools(bm.HoverTool(tooltips=[(key, "@" + key) for key in kwargs.keys()]))
    if show:
        pl.show(fig)
    return fig


def get_tsne_projection(word_vectors):
    tsne = TSNE(n_components=2)
    return tsne.fit_transform(word_vectors)