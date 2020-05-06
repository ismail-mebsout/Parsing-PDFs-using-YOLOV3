#%%
import pandas as pd
import numpy as np 
import camelot
from date_extractor import extract_dates
from dateparser.search import search_dates
import datetime
from itertools import groupby


 # %%
def eval_(x):
    try:
        return eval(x)
    except:
        return x

def mask(df, f):# lambda x:f(x)
    db_copy=df.copy()
    matrix_mask=[]
    for i in range(db_copy.shape[0]):
        row_mask=db_copy.iloc[i].apply(lambda x:f(x)).tolist()
        matrix_mask.append([int(x) for x in row_mask])
    return matrix_mask

def preproc(df):
    db_copy=df.copy()
    for i in range(db_copy.shape[1]):
        db_copy.iloc[:][i] = (
            db_copy.iloc[:][i].str.replace(",", "").apply(eval_)
        )
    return db_copy

def is_date(x):

    x=x.lower()
    month_regex = "^(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)|dec(?:ember)?"
    year_regex = "20[\d]{2}"
    date_regex = month_regex + "|" + year_regex
    
    line =pd.Series({0:x})
    
    row_mask_reg=(line.str.contains(date_regex) == True).tolist()[0]
    row_mask_f=isinstance(search_dates(x), list)

    # joint_res=np.multiply(row_mask_reg, row_mask_f)
    joint_res=row_mask_reg & row_mask_f

    return joint_res

def noise_deleting(db, threshold_emp=0.66, threshold_row=0.3 ,threshold_str=25):
    rows_to_drop = []
    for i in range(len(db)):
        if db.iloc[i][0] != "":
            if db.iloc[i].value_counts().idxmax() == "":
                if db.iloc[i].value_counts().max() / db.shape[1] > threshold_emp:
                    rows_to_drop.append(i)
                    
        if (db.iloc[i].apply(lambda x:len(str(x))).max()>threshold_str):
            if (db.iloc[i].apply(lambda x:len(str(x))).max()>2*threshold_str):
                rows_to_drop.append(i)
            elif (not any([is_date(str(x)) for x in db.iloc[i].tolist()])) and (not any([isinstance(eval_(x), int) for x in db.iloc[i].tolist()])):
                rows_to_drop.append(i)
    n=int(db.shape[0]*threshold_row)
    #rows_to_drop=[x for x in rows_to_drop if x<n]

    db = db.drop(rows_to_drop)
    db = db.reset_index(drop=True)

    return db

def merging_head(db, threshold=0.25, threshold_row=0.22):
    
    db_copy = preproc(db).copy()
    # for i in range(db_copy.shape[1]):
    #     db_copy.iloc[:][i] = (
    #         db_copy.iloc[:][i].str.replace(",", "").replace(" ", "").apply(eval_)
    #     )

    per_str_row = [
        db_copy.iloc[i].apply(lambda x: isinstance(x, str)).sum()
        / db_copy.shape[0]
        for i in range(db_copy.shape[0])
    ]
    potential_to_merge = [i for i, x in enumerate(per_str_row) if x >= threshold]

    has_date=[i for i in range(db_copy.shape[0]) if any([is_date(str(x)) for x in db_copy.iloc[i].tolist()])]
    has_money=[i for i in range(db_copy.shape[0]) if any([any(k in["£", "€", "$"] for k in list(str(x))) for x in db_copy.iloc[i].tolist()])]
    to_merge=[x for x in potential_to_merge if (x not in has_date) and (x not in has_money)]
    
    n=int(db_copy.shape[0]*threshold_row)
    to_merge=[x for x in to_merge if x<n]
    to_merge.sort()
    
    if to_merge:
        if to_merge[0]>0:
            bf_df=db.iloc[:to_merge[0]]
        
        header = db.iloc[to_merge[0]]
        to_merge.remove(to_merge[0])
        if len(to_merge)>=1:
            for idx in to_merge:
                header = header + " " + db.iloc[idx]
        header=pd.DataFrame(header).transpose()

        rest_bf = db.iloc[max(to_merge) + 1 :]
        try:
            new_base=pd.concat([bf_df, header, rest_bf])
        except UnboundLocalError:
            new_base=pd.concat([header, rest_bf])            

        return new_base

    return db

def extract_year(dataframe, keep_after=2016):
    db=dataframe.copy()
    current_year = datetime.datetime.now().year
    year = db[db.columns[0]].apply(lambda x: int(search_dates(x)[0][1].year) if search_dates(x) else 0)
    
    #year correction
    new_year=[x if (x<current_year) & (x>=keep_after) else 'nan' for x in year]

    direction=[max(x) for x in [list(v) for k, v in groupby(new_year, key='nan'.__ne__) if k]]
    if len(direction)==1:
        method="double"
    elif len(direction)>1:
        if direction[:2].index(min(direction[:2]))==0:
            method="ffill"
        else:
            method="bfill"

    new_year=[x if isinstance(x,int) else np.nan for x in new_year]
    if method!="double":
        new_year=pd.Series(new_year).fillna(method=method).values 
    elif method=="double":
        new_year=pd.Series(new_year).fillna(method="ffill").fillna(method="bfill").values 

    db["year"] = new_year
    # db = db[db.year >= keep_only]
    cols = db.columns.tolist()
    new_cols = [cols[-1]] + cols[:-1]

    return db[new_cols]

def is_double_index(df):
    date_mask=mask(df, is_date)
    flag= any(np.sum(np.array(date_mask), axis=0)>0) & (not np.sum(np.array(date_mask), axis=0)[0]>0)
    return 

def is_date_in_line(df):
    date_mask=mask(df, is_date)
    flag=np.sum(np.array(date_mask), axis=0)[0]>0
    return flag

def is_triple_index(df):
    """
    
    """





#%%



