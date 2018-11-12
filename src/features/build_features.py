import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def awesome_function(s):
    from IPython.display import display, HTML
    css = """
        .blink {
            animation-duration: 1s;
            animation-name: blink;
            animation-iteration-count: infinite;
            animation-timing-function: steps(2, start);
        }
        @keyframes blink {
            80% {
                visibility: hidden;
            }
        }"""

    to_show = HTML(
        '<style>{}</style>'.format(css) +
        '<p class="blink"> {} IS AWESOME!!!!! </p>'.format(s)
    )
    display(to_show)


def remove_invalid_data(path, path2):
    """ 
    Takes a path to a water pumps csv, loads in pandas, removes
    invalid columns and returns the dataframe.
    """
    # load data
    df_raw = pd.read_csv(path, index_col=0)
    lb = pd.read_csv(path2, index_col=0)

    df = pd.concat((df_raw, lb), axis=1)

    # preselected columns
    useful_columns = ['amount_tsh',
                      'date_recorded',
                      'gps_height',
                      'population',
                      'longitude',
                      'latitude',
                      'wpt_name',
                      'basin',
                      'construction_year',
                      'permit',
                      'payment_type',
                      'water_quality',
                      'quantity',
                      'management',
                      'extraction_type_class',
                      'payment',
                      'quality_group',
                      'source_type',
                      'waterpoint_type', 
                      'status_group']

    df = df[useful_columns]

    invalid_values = {
        'amount_tsh': {0: np.nan},
        'longitude': {0: np.nan},
        'population': {0: np.nan},
        'construction_year': {0: np.nan}
    }

    # drop rows with invalid values
    df.replace(invalid_values, inplace=True)

    # drop any rows in the dataset that have NaNs
    df = df.dropna(how="any")

    # create categorical columns
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = df[c].astype('category')

    return df, df_raw


def remove_columns(path):
    """ Takes a path to a water pumps csv, loads in pandas, removes
        invalid columns and returns the dataframe.
    """
    df = pd.read_csv(path, index_col=0)

    # preselected columns
    useful_columns = ['amount_tsh',
                      'date_recorded',
                      'gps_height',
                      'population',
                      'longitude',
                      'latitude',
                      'wpt_name',
                      'basin',
                      'construction_year',
                      'permit',
                      'payment_type',
                      'water_quality',
                      'quantity',
                      'management',
                      'extraction_type_class',
                      'payment',
                      'quality_group',
                      'source_type',
                      'waterpoint_type']

    df = df[useful_columns]
    
    df['water_quality'][df['water_quality'] == 'fluoride abandoned'] = 'fluoride'
    df['waterpoint_type'][df['waterpoint_type'] == 'dam'] = 'other'

    # create categorical columns
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = df[c].astype('category')

    return df


# Encode labels
def encode_label(label):
    return pd.Series(label).map({
        'non functional': 2,
        'functional needs repair': 1,
        'functional': 0
    })


# Decode labels
def decode_label(label):
    return pd.Series(label).map({
        2: 'non functional',
        1: 'functional needs repair',
        0: 'functional'
    })

class type_selector(BaseEstimator, TransformerMixin):
    '''
    Select type of feature.
    '''
    def __init__(self, dtype):
        self.dtype = dtype
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        assert isinstance(x, pd.DataFrame)
        
        return x.select_dtypes(include=[self.dtype])
    
    
class string_indexer(BaseEstimator, TransformerMixin):
    '''
    Simple indexer to handle missing values.
    '''
    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        assert isinstance(x, pd.DataFrame)
        
        return x.apply(lambda s: s.cat.codes.replace({-1: len(s.cat.categories)}))
    
    
class add_features(BaseEstimator, TransformerMixin):
    '''
    Add features
    '''
    def __init__(self, columns):
        self.columns = columns
        self.dicts = {}
        
    def fit(self, x, y=None):
        df = x.copy()
        df[y.name] = y
        
        for c in df.columns.tolist():
            if c in self.columns:
                # SIZE
                size = df.groupby(c)[y.name].size()
                size = size/len(df)
                self.dicts[c] = size
        
        df = df.drop([y.name], axis=1)
        return self
    
    def transform(self, x):
        assert isinstance(x, pd.DataFrame)
        
        for c in x.columns.tolist():
            if c in self.columns:
                # SIZE
                size = self.dicts[c]
                x[c +'_encode'] = x[c].map(size)
        
        return x