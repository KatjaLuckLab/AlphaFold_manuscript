#!/usr/bin/env python
# coding: utf-8

# In[1]:


#connection to server
import db_utils

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq
from matplotlib import gridspec
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')

connect = db_utils.get_connection()
cursor = connect.cursor()


# In[2]:


#Functions scripted by Katja
import pandas
def get_saturation_curve_query(project_id,NL_id,mCit_id,mCit_id_bleedthrough):
    query = f"""select a.NL_plasmid,a.mCit_plasmid,a.NL_property,a.mCit_property,
        a.avg_BRET_ratio-g.avg_BRET_ratio avg_BRET_ratio,
        (sqrt(power(a.std_BRET_ratio,2)+power(g.std_BRET_ratio,2))) std_BRET_ratio, 
        a.NL_plasmid_id,a.mCit_plasmid_id, 
        avg((d.avg_FL-f.avg_replicates)/(c.measurement-e.avg_replicates)) avg_expr_ratio, 
        std((d.avg_FL-f.avg_replicates)/(c.measurement-e.avg_replicates)) std_expr_ratio, 
        avg(d.avg_FL-f.avg_replicates) avg_FL_corr, 
        avg(c.measurement-e.avg_replicates) avg_LU_corr, 
        std(d.avg_FL-f.avg_replicates) std_FL_corr, 
        std(c.measurement-e.avg_replicates) std_LU_corr, 
        a.project_id 
        from luthy_data.BRET_ratios a,
        luthy_data.plate_layout b,
        luthy_data.LU_raw c,
        luthy_data.FL_avg_points d,
        luthy_data.avg_replicates e,
        luthy_data.avg_replicates f, 
        luthy_data.BRET_ratios g
        where a.project_id='{project_id}' and a.NL_plasmid_id='{NL_id}' and a.mCit_plasmid_id='{mCit_id}' 
        and a.NL_plasmid_id=b.NL_plasmid_id and a.mCit_plasmid_id=b.mCit_plasmid_id and a.project_id=b.project_id 
        and a.NL_property=b.NL_property and a.mCit_property=b.mCit_property and a.project_id=c.project_id and 
        c.measurement_id='totLu01' and b.well_id=c.well_id and a.project_id=d.project_id and 
        d.measurement_id='Fl01' and b.well_id=d.well_id and c.well_id=d.well_id and a.project_id=e.project_id and 
        e.NL_plasmid_id='KL_01' and e.mCit_plasmid_id='empty' and e.measurement_id=c.measurement_id and 
        a.project_id=f.project_id and e.NL_plasmid_id=f.NL_plasmid_id and e.mCit_plasmid_id=f.mCit_plasmid_id and 
        f.measurement_id=d.measurement_id and g.project_id=a.project_id and g.NL_plasmid_id='KL_03' and 
        g.mCit_plasmid_id='{mCit_id_bleedthrough}' and g.acc_measurement_id='accLu01' and b.include = 1 and 
        b.plate_id=c.plate_id and b.plate_id=d.plate_id
        group by a.NL_plasmid_id,a.mCit_plasmid_id,a.NL_property,a.mCit_property"""
    return query

def get_saturation_curve_query_fixedBT(project_id,NL_id,mCit_id,mCit_id_bleedthrough):
    query = f"""select a.NL_plasmid,a.mCit_plasmid,a.NL_property,a.mCit_property,
        a.avg_BRET_ratio-g.avg_BRET_ratio avg_BRET_ratio,
        (sqrt(power(a.std_BRET_ratio,2)+power(g.std_BRET_ratio,2))) std_BRET_ratio, 
        a.NL_plasmid_id,a.mCit_plasmid_id, 
        avg((d.avg_FL-f.avg_replicates)/(c.measurement-e.avg_replicates)) avg_expr_ratio, 
        std((d.avg_FL-f.avg_replicates)/(c.measurement-e.avg_replicates)) std_expr_ratio, 
        avg(d.avg_FL-f.avg_replicates) avg_FL_corr, 
        avg(c.measurement-e.avg_replicates) avg_LU_corr, 
        std(d.avg_FL-f.avg_replicates) std_FL_corr, 
        std(c.measurement-e.avg_replicates) std_LU_corr, 
        a.project_id 
        from luthy_data.BRET_ratios a,
        luthy_data.plate_layout b,
        luthy_data.LU_raw c,
        luthy_data.FL_avg_points d,
        luthy_data.avg_replicates e,
        luthy_data.avg_replicates f, 
        luthy_data.BRET_ratios g
        where a.project_id='{project_id}' and a.NL_plasmid_id='{NL_id}' and a.mCit_plasmid_id='{mCit_id}' 
        and a.NL_plasmid_id=b.NL_plasmid_id and a.mCit_plasmid_id=b.mCit_plasmid_id and a.project_id=b.project_id 
        and a.NL_property=b.NL_property and a.mCit_property=b.mCit_property and a.project_id=c.project_id and 
        c.measurement_id='totLu01' and b.well_id=c.well_id and a.project_id=d.project_id and 
        d.measurement_id='Fl01' and b.well_id=d.well_id and c.well_id=d.well_id and a.project_id=e.project_id and 
        e.NL_plasmid_id='KL_01' and e.mCit_plasmid_id='empty' and e.measurement_id=c.measurement_id and 
        a.project_id=f.project_id and e.NL_plasmid_id=f.NL_plasmid_id and e.mCit_plasmid_id=f.mCit_plasmid_id and 
        f.measurement_id=d.measurement_id and g.project_id='Lu128r01' and g.NL_plasmid_id='KL_03' and 
        g.mCit_plasmid_id='{mCit_id_bleedthrough}' and g.acc_measurement_id='accLu01' and b.include = 1 and 
        b.plate_id=c.plate_id and b.plate_id=d.plate_id
        group by a.NL_plasmid_id,a.mCit_plasmid_id,a.NL_property,a.mCit_property"""
    return query


def get_df_ppi_rand_pos(ppis,project_id_names,pos_ctrls,rand_ctrls_NL,mCit_protein_names,
                        mCit_proteins,pos_ctrl_NL_proteins,NL_protein_names,rand_ctrls_mCit,
                        pos_ctrl_mCit_proteins, NL_proteins):

    db_table = 'luthy_data.cBRET_ratios'

    data_df = pandas.DataFrame(columns=['pos','avg_BRET','std_BRET','type',
                                        'NL_plasmid_id','mCit_plasmid_id','project_id'])

    for i,ppi in enumerate(ppis):
        ppi_query = f"""select avg_cBRET_ratio,std_cBRET_ratio,NL_plasmid_id,mCit_plasmid_id,project_id 
                        from luthy_data.cBRET_ratios
                        where project_id in {project_id_names} and 
                        NL_property='1ng' and mCit_property='50ng' and NL_plasmid_id='{ppi[0]}' and 
                        mCit_plasmid_id='{ppi[1]}'"""
        cursor.execute(ppi_query)
        rows = cursor.fetchall()
        for row in rows:
            avg_bret = float(row[0])
            std_bret = float(row[1])
            data_df = data_df.append({'pos':i, 'avg_BRET':avg_bret, 'std_BRET':std_bret, 'type':'ppi', 
                                      'NL_plasmid_id':row[2],'mCit_plasmid_id':row[3],'project_id':row[4]},
                                     ignore_index=True)

    for i,ppi in enumerate(pos_ctrls):
        ppi_query = f"""select avg_cBRET_ratio,std_cBRET_ratio,NL_plasmid_id,mCit_plasmid_id,project_id 
                        from luthy_data.cBRET_ratios
                        where project_id in {project_id_names} and 
                        NL_property='1ng' and mCit_property='50ng' and NL_plasmid_id='{ppi[0]}' and 
                        mCit_plasmid_id='{ppi[1]}'"""
        cursor.execute(ppi_query)
        rows = cursor.fetchall()
        for row in rows:
            avg_bret = float(row[0])
            std_bret = float(row[1])
            data_df = data_df.append({'pos':i+len(ppis), 'avg_BRET':avg_bret, 'std_BRET':std_bret, 'type':'pos_ctrl', 
                                      'NL_plasmid_id':row[2],'mCit_plasmid_id':row[3],'project_id':row[4]},
                                     ignore_index=True)

    NL_rand_query = f"""select NL_plasmid_id,mCit_plasmid_id,avg_cBRET_ratio,std_cBRET_ratio,project_id 
                    from luthy_data.cBRET_ratios
                    where project_id in {project_id_names} and 
                    NL_property='1ng' and mCit_property='50ng' and NL_plasmid_id in {rand_ctrls_NL} and 
                    mCit_plasmid_id in {mCit_protein_names}"""
    cursor.execute(NL_rand_query)
    rows = cursor.fetchall()
    for row in rows:
        NL_id = row[0]
        mCit_id = row[1]
        avg_bret = float(row[2])
        std_bret = float(row[3])
        for j, plasmid_id in enumerate(mCit_proteins):
            if plasmid_id == mCit_id:
                data_df = data_df.append({'pos':j, 'avg_BRET':avg_bret, 'std_BRET':std_bret, 'type':'rand_ctrl', 
                                      'NL_plasmid_id':row[0],'mCit_plasmid_id':row[1],'project_id':row[4]},ignore_index=True)
        data_df = data_df.append({'pos':pos_ctrl_NL_proteins.index(NL_id)+len(ppis), 'avg_BRET':avg_bret, 'std_BRET':std_bret, 'type':'rand_ctrl', 
                                      'NL_plasmid_id':row[0],'mCit_plasmid_id':row[1],'project_id':row[4]},ignore_index=True)

    mCit_rand_query = f"""select NL_plasmid_id,mCit_plasmid_id,avg_cBRET_ratio,std_cBRET_ratio,project_id 
                    from luthy_data.cBRET_ratios
                    where project_id in {project_id_names} and 
                    NL_property='1ng' and mCit_property='50ng' and NL_plasmid_id in {NL_protein_names} and 
                    mCit_plasmid_id in {rand_ctrls_mCit}"""
    cursor.execute(mCit_rand_query)
    rows = cursor.fetchall()
    for row in rows:
        NL_id = row[0]
        mCit_id = row[1]
        avg_bret = float(row[2])
        std_bret = float(row[3])
        for j, plasmid_id in enumerate(NL_proteins):
            if plasmid_id == NL_id:
                data_df = data_df.append({'pos':j, 'avg_BRET':avg_bret, 'std_BRET':std_bret, 'type':'rand_ctrl', 
                                      'NL_plasmid_id':row[0],'mCit_plasmid_id':row[1],'project_id':row[4]},ignore_index=True)
        data_df = data_df.append({'pos':pos_ctrl_mCit_proteins.index(mCit_id)+len(ppis), 'avg_BRET':avg_bret, 'std_BRET':std_bret, 'type':'rand_ctrl', 
                                      'NL_plasmid_id':row[0],'mCit_plasmid_id':row[1],'project_id':row[4]},ignore_index=True)
        
    return data_df


def get_cBRET_expr_ratio_df(ppis,project_id_names,pos_ctrls,rand_ctrls_NL,rand_ctrls_mCit,NL_protein_names,
                            mCit_protein_names):

    data_df = pandas.DataFrame(columns=['NL_plasmid_id','mCit_plasmid_id','project_id','avg_cBRET_ratio',
                                        'std_cBRET_ratio','avg_expr_ratio','std_expr_ratio','type'])

    for i,ppi in enumerate(ppis):
        query = f"""select a.NL_plasmid_id,a.mCit_plasmid_id,a.project_id,a.avg_cBRET_ratio,a.std_cBRET_ratio,
            avg((d.avg_FL-f.avg_replicates)/(c.measurement-e.avg_replicates)) avg_expr_ratio, 
            std((d.avg_FL-f.avg_replicates)/(c.measurement-e.avg_replicates)) std_expr_ratio, 'ppi' type
            from luthy_data.cBRET_ratios a, 
            luthy_data.plate_layout b,
            luthy_data.LU_raw c,
            luthy_data.FL_avg_points d,
            luthy_data.avg_replicates e,
            luthy_data.avg_replicates f
            where a.project_id in {project_id_names} and a.NL_plasmid_id='{ppi[0]}' and a.mCit_plasmid_id='{ppi[1]}' 
            and a.NL_property='1ng' and a.mCit_property='50ng' and a.project_id=b.project_id and 
            a.NL_plasmid_id=b.NL_plasmid_id and a.mCit_plasmid_id=b.mCit_plasmid_id and a.NL_property=b.NL_property 
            and a.mCit_property=b.mCit_property and b.project_id=c.project_id and 
            c.measurement_id='totLu01' and b.well_id=c.well_id and b.project_id=d.project_id and 
            d.measurement_id='Fl01' and b.well_id=d.well_id and c.well_id=d.well_id and b.project_id=e.project_id and 
            e.NL_plasmid_id='KL_01' and e.mCit_plasmid_id='empty' and e.measurement_id=c.measurement_id and 
            b.project_id=f.project_id and e.NL_plasmid_id=f.NL_plasmid_id and e.mCit_plasmid_id=f.mCit_plasmid_id and 
            f.measurement_id=d.measurement_id
            group by a.NL_plasmid_id,a.mCit_plasmid_id,a.project_id"""
        expr_df = pandas.read_sql(query,connect)
        data_df = data_df.append(expr_df,ignore_index=True)

    for i,ppi in enumerate(pos_ctrls):
        query = f"""select a.NL_plasmid_id,a.mCit_plasmid_id,a.project_id,a.avg_cBRET_ratio,a.std_cBRET_ratio,
            avg((d.avg_FL-f.avg_replicates)/(c.measurement-e.avg_replicates)) avg_expr_ratio, 
            std((d.avg_FL-f.avg_replicates)/(c.measurement-e.avg_replicates)) std_expr_ratio, 'pos_ctrl' type
            from luthy_data.cBRET_ratios a, 
            luthy_data.plate_layout b,
            luthy_data.LU_raw c,
            luthy_data.FL_avg_points d,
            luthy_data.avg_replicates e,
            luthy_data.avg_replicates f
            where a.project_id in {project_id_names} and a.NL_plasmid_id='{ppi[0]}' and a.mCit_plasmid_id='{ppi[1]}' 
            and a.NL_property='1ng' and a.mCit_property='50ng' and a.project_id=b.project_id and 
            a.NL_plasmid_id=b.NL_plasmid_id and a.mCit_plasmid_id=b.mCit_plasmid_id and a.NL_property=b.NL_property 
            and a.mCit_property=b.mCit_property and b.project_id=c.project_id and 
            c.measurement_id='totLu01' and b.well_id=c.well_id and b.project_id=d.project_id and 
            d.measurement_id='Fl01' and b.well_id=d.well_id and c.well_id=d.well_id and b.project_id=e.project_id and e.NL_plasmid_id='KL_01' and e.mCit_plasmid_id='empty' and e.measurement_id=c.measurement_id and 
            b.project_id=f.project_id and e.NL_plasmid_id=f.NL_plasmid_id and e.mCit_plasmid_id=f.mCit_plasmid_id and 
            f.measurement_id=d.measurement_id 
            group by a.NL_plasmid_id,a.mCit_plasmid_id,a.project_id"""
        expr_df = pandas.read_sql(query,connect)
        data_df = data_df.append(expr_df,ignore_index=True)

    query = f"""select a.NL_plasmid_id,a.mCit_plasmid_id,a.project_id,a.avg_cBRET_ratio,a.std_cBRET_ratio,
        avg((d.avg_FL-f.avg_replicates)/(c.measurement-e.avg_replicates)) avg_expr_ratio, 
        std((d.avg_FL-f.avg_replicates)/(c.measurement-e.avg_replicates)) std_expr_ratio, 'rand_ctrl' type
        from luthy_data.cBRET_ratios a, 
        luthy_data.plate_layout b,
        luthy_data.LU_raw c,
        luthy_data.FL_avg_points d,
        luthy_data.avg_replicates e,
        luthy_data.avg_replicates f
        where a.project_id in {project_id_names} and ((a.NL_plasmid_id in {rand_ctrls_NL} and 
        a.mCit_plasmid_id in {mCit_protein_names} ) or (a.NL_plasmid_id in {NL_protein_names} and 
        a.mCit_plasmid_id in {rand_ctrls_mCit}))
        and a.NL_property='1ng' and a.mCit_property='50ng' and a.project_id=b.project_id and 
        a.NL_plasmid_id=b.NL_plasmid_id and a.mCit_plasmid_id=b.mCit_plasmid_id and a.NL_property=b.NL_property 
        and a.mCit_property=b.mCit_property and b.project_id=c.project_id and 
        c.measurement_id='totLu01' and b.well_id=c.well_id and b.project_id=d.project_id and 
        d.measurement_id='Fl01' and b.well_id=d.well_id and and c.well_id=d.well_id and b.project_id=e.project_id and 
        e.NL_plasmid_id='KL_01' and e.mCit_plasmid_id='empty' and e.measurement_id=c.measurement_id and 
        b.project_id=f.project_id and e.NL_plasmid_id=f.NL_plasmid_id and e.mCit_plasmid_id=f.mCit_plasmid_id and 
        f.measurement_id=d.measurement_id  
        group by a.NL_plasmid_id,a.mCit_plasmid_id,a.project_id"""
    expr_df = pandas.read_sql(query,connect)
    data_df = data_df.append(expr_df,ignore_index=True)
    return data_df
    


# In[ ]:




