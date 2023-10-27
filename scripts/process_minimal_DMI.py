# Author: Chop Yan Lee
import calculate_template_independent_metrics
import calculate_template_dependent_metrics
import os, argparse
from pymol import cmd
import pandas as pd

DMI_structure_dict = {}

class Prediction_folder(calculate_template_independent_metrics.Prediction_folder):
    """Class that stores prediction folder information"""
    def __init__(self,prediction_folder,num_model,dmi_name,project_name):
        super().__init__(prediction_folder=prediction_folder,num_model=num_model,project_name=project_name)
        self.dmi_name = dmi_name
        self.run_id = self.prediction_name.split('_')[0]

    def assign_dmi_name(self):
        """Assign DMI name information to all Predicted_model instance
        """
        for model_id, model_inst in self.model_instances.items():
            model_inst.dmi_name = self.dmi_name

    def instantiate_predicted_model(self):
        """Override the method from parent class so that the function instantiate Predicted_model using the inherited class from this script
        """
        self.model_instances = {f'ranked_{i}':Predicted_model(f'ranked_{i}') for i in range(self.num_model)}

    def process_all_models(self):
        """Override the method of parent class to include template dependent metric processing"""
        self.parse_ranking_debug_file()
        self.parse_prediction_fasta_file()
        if self.predicted:
            self.instantiate_predicted_model()
            self.assign_dmi_name()
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
            'RMSD_domain': float,
            'num_align_atoms_domain': float,
            'align_score_domain': float,
            'num_align_resi_domain': float,
            'RMSD_backbone_peptide': float,
            'RMSD_all_atom_peptide': float,
            'known_motif_plddt': float,
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
            row = common_info + ['Prediction failed'] + [None]*22
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
    def __init__(self,predicted_model):
        super().__init__(predicted_model)
        self.RMSD_domain = None
        self.num_align_atoms_domain = None
        self.align_score_domain = None
        self.num_align_resi_domain = None
        self.RMSD_backbone_peptide = None
        self.RMSD_all_atom_peptide = None
        self.DockQ = None
        self.Fnat = None
        self.iRMS = None
        self.LRMS = None
        self.Fnonnat = None
        self.dmi_name = None
        self.DMI_structure_inst = None
    
    def assign_DMI_structure_inst(self):
        """Takes the previously read-in DMI structures and use the self.dmi_name to find the DMI_structure instance. Then assign the DMI_structure instance to the self.DMI_structure_inst attribute. Source the global variable DMI_structure_dict for the assignment
        """
        global DMI_structure_dict
        self.DMI_structure_inst = DMI_structure_dict.get(self.dmi_name)

    def calculate_template_dependent_metrics(self):
        """Wrapper function to calculate template dependent metric of a predicted model
        """
        model_path = os.path.join(self.path_to_model,f'{self.predicted_model}.pdb')
        self.RMSD_domain, self.num_align_atoms_domain,self.align_score_domain,self.num_align_resi_domain,self.RMSD_backbone_peptide,self.RMSD_all_atom_peptide = calculate_template_dependent_metrics.process_calculate_model_RMSD(DMI_structure_inst=self.DMI_structure_inst,predicted_model=model_path)
        self.save_superimposed_model()
        cmd.reinitialize()
        DockQ_metrics = calculate_template_dependent_metrics.calculate_DockQ(template_model=f'{self.DMI_structure_inst.download_directory}{self.DMI_structure_inst.pdb_id}_min_DMI.pdb',predicted_model=os.path.join(self.path_to_model,f'{self.predicted_model}_min.pdb'))
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
        self.get_model_independent_metrics()
        self.assign_DMI_structure_inst()
        self.calculate_template_dependent_metrics()
        print(f'{os.path.join(self.path_to_model,self.predicted_model)} processed!')

def parse_prediction_name(prediction_name):
    """Parse out different components in the standard name for a minimal DMI prediction (e.g. run37_DEG_APCC_KENBOX_2_4GGD)

    Args:
        prediction_name (str): Name of the folder containing the AlphaFold predicted structure

    Returns:
        run_id (str): Run ID
        dmi_name (str): Name of DMI type
        pdb_id (str): PDB ID
    """
    splits = prediction_name.split('_')
    run_id = splits[0]
    dmi_name = '_'.join(splits[1:-1])
    pdb_id = splits[-1]
    return run_id, dmi_name, pdb_id

def main():
    """Parse arguments and wraps all functions into main for executing the program in a way that it can handle multiple run ids given to it
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-run_ids', type=str, help='Run IDs for metrics calculation', dest='run_ids')
    parser.add_argument('-path_to_run', type=str, help='Path to the run folder "/" at the end', dest='path_to_run')
    args = parser.parse_args()
    run_ids = vars(args)['run_ids']
    path_to_run = vars(args)['path_to_run']
    calculated_dmi = []
    global DMI_structure_dict
    DMI_structure_dict = calculate_template_dependent_metrics.read_in_annotated_DMI_structure('/Volumes/imb-luckgr/projects/dmi_predictor/DMI_AF2_PRS/AF2_DMI_structure_PRS - Sheet1.tsv')
    pdb_ids = [dmi.pdb_id for dmi in DMI_structure_dict.values()]
    for run_id in run_ids.split(','):
        if os.path.exists(f'{path_to_run}run{run_id}/run{run_id}_template_indep_dep_info.tsv'):
            temp = pd.read_csv(f'{path_to_run}run{run_id}/run{run_id}_template_indep_dep_info.tsv',sep='\t',index_col=0)
            calculated_dmi = temp['prediction_name'].unique()
        run_path = f'{path_to_run}run{run_id}'
        for file in os.listdir(run_path):
            if file in calculated_dmi:
                # print(file)
                continue
            file_abs = os.path.join(run_path,file)
            if os.path.isdir(file_abs):
                _, dmi_name, pdb_id = parse_prediction_name(file)
                if dmi_name not in DMI_structure_dict: # for the processing of older runs where some dmi_types were removed
                    continue
                if pdb_id not in pdb_ids:
                    continue
                folder = Prediction_folder(file_abs,num_model=5,dmi_name=dmi_name,project_name='AlphaFold_benchmark')
                folder.process_all_models()
                folder.write_out_calculated_metrics()

if __name__ == '__main__':
    main()

# python3 /Users/chopyanlee/AlphaFold_benchmark/scripts/process_minimal_DMI.py -run_ids 48 -path_to_run /Volumes/imb-luckgr/projects/dmi_predictor/DMI_AF2_PRS/