from comparewikilist import similar
import pandas as pd
import numpy as np

def get_df(store):
    print(store)
    # Build dataframe based on all permutations of topics
    for i in range(len(store)):
        article1 = store[i]
        article2list = store[i+1:]
        if i == 0:
            df = pd.DataFrame(np.array(similar(article1,article2list, verbose=0).iterate()))
        else:
            df = pd.concat([df, pd.DataFrame(np.array(similar(article1,article2list, verbose=0).iterate()))], axis=0)

    df.columns = ['Topic 1', 'Topic 2', 'Probability', 'Similar?']
    df.index = range(df.shape[0])
    df['Probability'] = df['Probability'].astype('int')
            
    return df