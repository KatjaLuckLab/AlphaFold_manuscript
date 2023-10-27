# This script is adapted from calculate_template_independent_metrics.py to calculate only pDockQ and iPAE of predicted models.
# Author: Chop Yan Lee

import calculate_template_independent_metrics
import os, argparse, sys, itertools, pickle
from pymol import cmd
import pandas as pd
import mdtraj as md
import numpy as np
from collections import defaultdict

class Prediction_folder(calculate_template_independent_metrics.Prediction_folder):
    """Class that stores prediction folder information"""
    def __init__(self,prediction_folder,num_model=5,project_name=None):
        super().__init__(prediction_folder=prediction_folder,num_model=num_model,project_name=project_name)
        self.run_id = self.prediction_name.split('_')[0]
        
    def write_out_calculated_metrics(self):
        """Override the same method from parent class. Write out the pDockQ and iPAE for every predicted model. Check if {run_id}_pDockQ_iPAE.tsv already exists, if it does, read in the file as pd.Dataframe and append new info into it, otherwise create a new one

        Returns:
            {run_id}_pDockQ_iPAE.tsv: A tsv file with the calculated pDockQ and iPAE metrics
        """
        metrics_out_path = os.path.join(self.path_to_prediction_folder,f'{self.run_id}_pDockQ_iPAE.tsv')
        # prepare the metrics dataframe
        metrics_columns_dtype = {
            'project_name':str,
            'prediction_name':str,
            'chainA_length':int,
            'chainB_length':int,
            'model_id':str,
            'pDockQ':float,
            'iPAE':float,
            }
        # check if template_independent_info.tsv already exists
        if os.path.exists(metrics_out_path):
            metrics_df = pd.read_csv(metrics_out_path,sep='\t',index_col=0)
            metrics_df.reset_index(drop=True,inplace=True)
        else:
            metrics_df = pd.DataFrame(columns=metrics_columns_dtype.keys())
            metrics_df = metrics_df.astype(dtype=metrics_columns_dtype)

        common_info = [self.project_name,self.prediction_name,len(self.fasta_sequence_dict.get('A')), len(self.fasta_sequence_dict.get('B'))]
        # check if the prediction folder has been predicted successfully without internal error from AlphaFold
        if not self.predicted:
            row = common_info + ['Prediction failed'] + [None]*2
            metrics_df.loc[len(metrics_df)] = row
        else:
            # insert metric info in a row-wise manner
            for model_id, model_inst in self.model_instances.items():
                row = common_info + [model_id]
                for column in list(metrics_columns_dtype)[5:]:
                    row.append(model_inst.__dict__.get(column))
                print(row)
                metrics_df.loc[len(metrics_df)] = row
        metrics_df.to_csv(metrics_out_path,sep='\t')
        print(f'Calculated metrics saved in {metrics_out_path}!')

    def instantiate_predicted_model(self):
        """Initialize the amount of Predicted_model instance according to the number of model specified and save it in the dict self.model_instances
        """
        self.model_instances = {f'ranked_{i}':Predicted_model(f'ranked_{i}') for i in range(self.num_model)}

    def process_all_models(self):
        """Use the instances of Predicted_model and run the wrapper function Predicted_model.get_model_independent_metrics function on themselves
        """
        self.parse_ranking_debug_file()
        self.parse_prediction_fasta_file()
        if self.predicted:
            self.instantiate_predicted_model()
            self.assign_model_info()
            for model_id, model_inst in self.model_instances.items():
                model_inst.calculate_pDockQ_iPAE()

class Predicted_model(calculate_template_independent_metrics.Predicted_model):
    """Class that stores predicted model"""
    def __init__(self,predicted_model):
        """Initialize an instance of Predicted_model
        
        Args:
            predicted_model (str): name of the predicted model like ranked_0
        """
        super().__init__(predicted_model)
        self.predicted_model = predicted_model
        self.path_to_model = None
        self.multimer_model = None
        self.chain_coords = None
        self.chain_plddt = None
        self.pickle_data = None
        self.pDockQ = np.nan
        self.PPV = np.nan
        self.iPAE = np.nan

    def calculate_pDockQ_iPAE(self):
        """Wraps the functions needed to calculate pDockQ and iPAE. Checks if the model is predicted by AF-MMv2.2 or v2.3. If it is v2.3, skip iPAE calculation due to JAX dependency.
        """
        if 'multimer_v2' in self.multimer_model:
            self.read_pickle()
            self.calculate_iPAE()
        self.calculate_pDockQ()

def main():
    """Parse arguments and wraps all functions into main for executing the program in such a way that it can handle multiple run ids given to it
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-run_ids', type=str, help='Run IDs for metrics calculation', dest='run_ids')
    parser.add_argument('-path_to_run', type=str, help='Either provide a path to a folder where multiple runs of AlphaFold predictions are contained and specify the run_ids to be processed or use -path_to_prediction to specify a folder that you want to process, include "/" at the end', dest='path_to_run')
    parser.add_argument('-path_to_prediction', type=str, help='Path to the prediction folder "/" at the end', dest='path_to_prediction')
    parser.add_argument('-project_name', type=str, help='Optional name for the project', dest='project_name')

    args = parser.parse_args()
    run_ids = vars(args)['run_ids']
    path_to_run = vars(args)['path_to_run']
    path_to_prediction = vars(args)['path_to_prediction']
    project_name = vars(args)['project_name']

    # a list to contains already processed files
    calculated_files = []

    # check which argument, -path_to_run or -path_to_prediction, is provided
    if (path_to_run is None) and (path_to_prediction is None):
        print('Please provide either -path_to_run or -path_to_prediction and try again!')
        sys.exit()
    elif project_name is None:
        print('Please input a project name!')
        sys.exit()
    elif path_to_prediction is not None:
        if os.path.exists(f'{path_to_prediction}pDockQ_iPAE.tsv'):
            temp = pd.read_csv(f'{path_to_prediction}pDockQ_iPAE.tsv',sep='\t',index_col=0)
            calculated_files = temp['prediction_name'].unique()
        for file in os.listdir(path_to_prediction):
            if file in calculated_files:
                # print(file)
                continue
            file_abs = os.path.join(path_to_prediction,file)
            if os.path.isdir(file_abs):
                folder = Prediction_folder(file_abs,num_model=5,project_name=project_name)
                folder.process_all_models()
                folder.write_out_calculated_metrics()
    else:
        for run_id in run_ids.split(','):
            if os.path.exists(f'{path_to_run}run{run_id}/run{run_id}_pDockQ_iPAE.tsv'):
                temp = pd.read_csv(f'{path_to_run}run{run_id}/run{run_id}_pDockQ_iPAE.tsv',sep='\t',index_col=0)
                calculated_files = temp['prediction_name'].unique()
            run_path = f'{path_to_run}run{run_id}'
            for file in os.listdir(run_path):
                if file in calculated_files:
                    continue
                file_abs = os.path.join(run_path,file)
                if os.path.isdir(file_abs):
                    folder = Prediction_folder(file_abs,num_model=5,project_name=project_name)
                    folder.process_all_models()
                    folder.write_out_calculated_metrics()

if __name__ == '__main__':
    main()