# Author: Chop Yan Lee
import calculate_template_independent_metrics
import DDI_calculate_template_dependent_metrics
import os, argparse, re
from pymol import cmd
import pandas as pd

class Prediction_folder(calculate_template_independent_metrics.Prediction_folder):
    """Class that stores prediction folder information"""
    def __init__(self,prediction_folder,num_model,project_name):
        super().__init__(prediction_folder=prediction_folder,num_model=num_model,project_name=project_name)
        self.run_id = self.prediction_name.split('_')[0]

    def parse_prediction_name(self):
        """Parse out different components in the standard name for a minimal DDI prediction (e.g. run5_PF00010_PF02344_1NKP_B_resi203_resi264.D_resi545_resi581)

        Args:
            prediction_name (str): Name of the folder containing the AlphaFold predicted structure

        Returns:
            ddi_name (str): the name of DDI type
            pdb_id (str): the PDB ID of the solved structure
            domain_chainA (str): the chain in the solved structure that shows to the first domain of the DDI type
            domain_startA (int): the residue index (resi) that denotes the start of the domain1 in the solved structure
            domain_endA (int): the residue index (resi) that denotes the end of the domain1 in the solved structure
            domain_chainB (str): the chain in the solved structure that shows to the second domain of the DDI type
            domain_startB (int): the residue index (resi) that denotes the start of the domain2 in the solved structure
            domain_endB (int): the residue index (resi) that denotes the end of the domain2 in the solved structure
        """
        ddi_name = re.search(r'(PF\d+_PF\d+)',self.prediction_name)
        ddi_name = ddi_name.group()
        pdb_id = self.prediction_name.split('_')[3].upper()
        structure_info = '_'.join(self.prediction_name.split('_')[4:])
        structure_info = re.sub(r'\.0','',structure_info)
        chainA_info, chainB_info = structure_info.split('.')
        domain_chainA, domain_startA, domain_endA = [re.sub(r'resi','',ele) for ele in chainA_info.split('_')]
        domain_chainB, domain_startB, domain_endB = [re.sub(r'resi','',ele) for ele in chainB_info.split('_')]

        return ddi_name,pdb_id,domain_chainA,domain_startA,domain_endA,domain_chainB,domain_startB,domain_endB

    def instantiate_predicted_model(self):
        """Override the method from parent class so that the function instantiate Predicted_model using the inherited class from this script
        """
        ddi_name,pdb_id,domain_chainA,domain_startA,domain_endA,domain_chainB,domain_startB,domain_endB = self.parse_prediction_name()
        self.model_instances = {f'ranked_{i}':Predicted_model(f'ranked_{i}',ddi_name,pdb_id,domain_chainA,domain_startA,domain_endA,domain_chainB,domain_startB,domain_endB) for i in range(self.num_model)}

    def process_all_models(self):
        """Override the method of parent class to include template dependent metric processing"""
        self.parse_ranking_debug_file()
        self.parse_prediction_fasta_file()
        if self.predicted:
            self.instantiate_predicted_model()
            self.assign_model_info()
            for model_id, model_inst in self.model_instances.items():
                model_inst.get_model_metrics()

    def write_out_calculated_metrics(self):
        """Override the same method from parent class. Write out the information that has been processed for every predicted model. Check if {run_id}_template_indep_dep_info.tsv already exists, if it does, read in the file as pd.Dataframe and append new info into it, otherwise create a new one

        Args:
            project_name (str): a project name given to model contacts dataframe as a key identifier

        Returns:
            {self.run_id}_template_indep_dep_info.tsv: A tsv file with the calculated template independent metrics
        """
        metrics_out_path = os.path.join(self.path_to_prediction_folder,f'{self.run_id}_template_indep_dep_info.tsv')
        # prepare the metrics dataframe
        metrics_columns_dtype = {
            'project_name':str,
            'prediction_name':str,
            'chainA_length':int,
            'chainB_length':int,
            'model_id':str,
            'model_confidence':float,
            'chainA_intf_avg_plddt':float,
            'chainB_intf_avg_plddt':float,
            'intf_avg_plddt':float,
            'pDockQ':float,
            'iPAE':float,
            'num_chainA_intf_res':int,
            'num_chainB_intf_res':int,
            'num_res_res_contact':int,
            'num_atom_atom_contact':int,
            'RMSD_big_domain' : float,
            'num_align_atoms_big_domain' : float,
            'align_score_big_domain' : float,
            'num_align_resi_big_domain' : float,
            'RMSD_backbone_small_domain' : float,
            'RMSD_all_atom_small_domain' : float,
            'DockQ': float,
            'Fnat': float,
            'iRMS': float,
            'LRMS': float,
            'Fnonnat': float,
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
            print(f'{self.prediction_folder} not predicted!')
            row = common_info + ['Prediction failed'] + [None]*21
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
    
    def write_out_contacts(self):
        """Write out atom-atom contacts into a tsv file
        
        Returns:
            {self.run_id}_all_model_contacts.tsv: A tsv file with the calculated information of atom-atom contact in all predicted models
        """
        contact_out_path = os.path.join(self.path_to_prediction_folder,f'{self.run_id}_all_model_contacts.tsv')
        # prepare the model contact dataframe
        contact_columns_dtype = {
            'project_name':str,
            'run_id':str,
            'prediction_name':str,
            'model_id':str,
            'chain1':str,
            'res1':str,
            'resID1':int,
            'atom1':str,
            'chain2':str,
            'res2':str,
            'resID2':int,
            'atom2':str,
            'atom_distance':float
            }
        # check if all_model_contacts.tsv already exists
        if os.path.exists(contact_out_path):
            contact_df = pd.read_csv(contact_out_path,sep='\t',index_col=0)
            contact_df.reset_index(drop=True,inplace=True)
        else:
            contact_df = pd.DataFrame(columns=contact_columns_dtype.keys())
            contact_df = contact_df.astype(dtype=contact_columns_dtype)
        
        # insert the contact info in a row-wise manner
        common_info = [self.project_name,self.run_id,self.prediction_name]
        for model_id, model_inst in self.model_instances.items():
            for atom_atom_contact in model_inst.atom_atom_contacts:
                row = common_info + [model_id] + atom_atom_contact
                contact_df.loc[len(contact_df)] = row
        contact_df.to_csv(contact_out_path,sep='\t')
        print(f'Calculated atom-atom contacts saved in {contact_out_path}!')

class Predicted_model(calculate_template_independent_metrics.Predicted_model):
    """Class that stores predicted model"""
    def __init__(self,predicted_model,ddi_name,pdb_id,domain_chainA,domain_startA,domain_endA,domain_chainB,domain_startB,domain_endB):
        super().__init__(predicted_model)
        self.ddi_name=ddi_name
        self.pdb_id=pdb_id
        self.domain_chainA=domain_chainA
        self.domain_startA=domain_startA
        self.domain_endA=domain_endA
        self.domain_chainB=domain_chainB
        self.domain_startB=domain_startB
        self.domain_endB=domain_endB
        self.RMSD_big_domain = None
        self.num_align_atoms_big_domain = None
        self.align_score_big_domain = None
        self.num_align_resi_big_domain = None
        self.RMSD_backbone_small_domain = None
        self.RMSD_all_atom_small_domain = None
        self.DockQ = None
        self.Fnat = None
        self.iRMS = None
        self.LRMS = None
        self.Fnonnat = None

    def check_chain_id(self):
        """Some models have their chain ids start from B instead of A. As the code requires the chain ids to be consistent (start from chain A), this function checks and rename the chain ids if necessary
        """
        model_path = os.path.join(self.path_to_model,f'{self.predicted_model}.pdb')
        cmd.load(model_path)
        chains = cmd.get_chains(f'{self.predicted_model}')
        if 'C' in chains:
            # change the chain id into a temporary arbitrary name
            cmd.alter(f'{self.predicted_model} and chain B', 'chain="tempA"')
            cmd.sort()
            cmd.alter(f'{self.predicted_model} and chain C', 'chain="tempB"')
            cmd.sort()
            # change the chain id into A and B
            cmd.alter(f'{self.predicted_model} and chain tempA', 'chain="A"')
            cmd.sort()
            cmd.alter(f'{self.predicted_model} and chain tempB', 'chain="B"')
            cmd.sort()
            # save and overwrite the predicted model
            cmd.save(model_path,f'{self.predicted_model}')
        cmd.reinitialize()

    def calculate_template_dependent_metrics(self):
        """Wrapper function to calculate template dependent metric of a predicted model
        """
        model_path = os.path.join(self.path_to_model,f'{self.predicted_model}.pdb')
        DDI_calculate_template_dependent_metrics.download_pdb(ddi_name=self.ddi_name,pdb_id=self.pdb_id)
        DDI_calculate_template_dependent_metrics.extract_minimal_DDI(ddi_name=self.ddi_name,pdb_id=self.pdb_id,domain_chainA=self.domain_chainA,domain_startA=self.domain_startA,domain_endA=self.domain_endA,domain_chainB=self.domain_chainB,domain_startB=self.domain_startB,domain_endB=self.domain_endB)
        template_model = f'/Volumes/imb-luckgr/projects/ddi_predictor/DDI_manual_curation/{self.ddi_name}/{self.pdb_id}_min_DDI.pdb'
        self.RMSD_big_domain, self.RMSD_all_atom_small_domain, self.RMSD_backbone_small_domain, self.num_align_atoms_big_domain, self.align_score_big_domain, self.num_align_resi_big_domain = DDI_calculate_template_dependent_metrics.perform_RMSD_calculation(predicted_model=model_path,template_model=template_model)
        self.save_superimposed_model()
        DockQ_metrics = DDI_calculate_template_dependent_metrics.calculate_DockQ(predicted_model=model_path,template_model=template_model)
        self.DockQ = DockQ_metrics.get('DockQ')
        self.Fnat = DockQ_metrics.get('Fnat')
        self.iRMS = DockQ_metrics.get('iRMS')
        self.LRMS = DockQ_metrics.get('LRMS')
        self.Fnonnat = DockQ_metrics.get('Fnonnat')

    def save_superimposed_model(self):
        """Save the superimposed model and reinitialize pymol session
        """
        cmd.save(os.path.join(self.path_to_model,f'{self.predicted_model}_superimpose.pse'))
        print(f'PyMol session of superimposed structure saved in {os.path.join(self.path_to_model,self.predicted_model)}_superimpose.pse!')
        cmd.reinitialize()

    def get_model_metrics(self):
        """Wraps the get_model_independent_metrics function of parent class to include template dependent metric calculation
        """
        self.check_chain_id()
        self.get_model_independent_metrics()
        self.calculate_template_dependent_metrics()
        print(f'{os.path.join(self.path_to_model,self.predicted_model)} processed!')

def main():
    """Parse arguments and wraps all functions into main for executing the program in a way that it can handle multiple run ids given to it
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-run_ids', type=str, help='Run IDs for metrics calculation', dest='run_ids')
    parser.add_argument('-path_to_run', type=str, help='Path to the run folder "/" at the end', dest='path_to_run')
    args = parser.parse_args()
    run_ids = vars(args)['run_ids']
    path_to_run = vars(args)['path_to_run']
    calculated_predictions = []

    for run_id in run_ids.split(','):
        if os.path.exists(f'{path_to_run}run{run_id}/run{run_id}_template_indep_dep_info.tsv'):
            temp = pd.read_csv(f'{path_to_run}run{run_id}/run{run_id}_template_indep_dep_info.tsv',sep='\t',index_col=0)
            calculated_predictions = temp['prediction_name'].unique()
        run_path = f'{path_to_run}run{run_id}'
        for file in os.listdir(run_path):
            if file in calculated_predictions:
                # print(file)
                continue
            file_abs = os.path.join(run_path,file)
            if os.path.isdir(file_abs):
                if not os.path.exists(os.path.join(run_path,f"{file}.fasta")):
                    print(f"Skipping the folder named {file}")
                    continue
                folder = Prediction_folder(file_abs,num_model=5,project_name='DDI_predictor')
                folder.process_all_models()
                folder.write_out_calculated_metrics()

if __name__ == '__main__':
    main()

# python3 /Users/chopyanlee/Coding/Python/DDI/DDI_predictor_AlphaFold/scripts/process_minimal_DDI.py -run_ids 5 -path_to_run /Users/chopyanlee/Coding/Python/DDI/AF_code_test/