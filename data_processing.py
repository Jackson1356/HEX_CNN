import os
import scanpy as sc
import pandas as pd
import numpy as np
import math


def data_processing(data_path, file_nums, k):
    labels = []
    nums = file_nums
    features = np.zeros((len(nums), k, 64, 78))
    for i in range(len(nums)):
        print(f'processing no. {nums[i]}')
        num = nums[i]
        cells=pd.read_csv(f'{data_path}/{num}/ck17_{num}_cell_barcodes.txt')
        gene=pd.read_csv(f'{data_path}/{num}/ck17_{num}_gene_names.txt', dtype={
            'no': 'int64',
            'gene name': 'string'
        })
        meta=pd.read_csv(f'{data_path}/{num}/ck17_{num}_metadata.txt')
        if 'barcode' not in meta.columns:
            meta=meta.rename(columns={"Unnamed: 0":"barcode"})
        adata=sc.read_mtx(f'{data_path}/{num}/ck17_{num}_gex_data.txt').T
        position = pd.read_csv(f'{data_path}/{num}/ck17_{num}_tissue_positions_list.csv', names=['in_tissue','row','col','pixel_row','pixel_col'])
        position['barcode'] = position.index
        adata.obs.index=cells['x']
        adata.var.index=gene['x']
        meta.index=meta.iloc[:,0]
        adata.obs=meta
    
        adata.obs.index.name='idx'
        obs = adata.obs.merge(position[['row', 'col', 'barcode']], on='barcode', how='inner')
        # get the label: 0 for non-responder and 1 for responder
        if 'ici_response' in obs.columns:
            if obs['ici_response'].unique().item() == 'NR':
                labels.append(0)
            else:
                labels.append(1)

        # get geneswith k-largest normalized gex
        gex_filter = np.asarray(np.sum(adata.X.todense()>0,axis=0)/2261)
        gene_no = np.argpartition(gex_filter, len(gex_filter) - k)
        gex_total = np.asarray(adata.X.todense())
    
        for j in range(k):
            for idx, row in obs.iterrows():
                hex_row = row['row']
                hex_col = row['col']
                mat_col = hex_row
                mat_row = math.floor(hex_col/2)
                features[i][j][mat_row][mat_col] = gex_total[idx][gene_no[0][-k:][j]]

    return features, labels


