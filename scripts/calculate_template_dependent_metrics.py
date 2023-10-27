# This script contains generic functions that calculates metrics from predicted models with the use of a template.
# Author: Chop Yan Lee
from pymol import cmd
import os, subprocess, ast, re, time
import pandas as pd
import numpy as np

class DMI_structure:
    """Class that store the annotated information of minimal region of DMIs in solved structures"""
    def __init__(self, dmi_name, pdb_id, domain_chain, domain_start, domain_end, motif_chain, motif_start, motif_end,download_directory=None):
        """Initialize an instance of DMI_structure
        
        Args:
            dmi_name (str): the name of DMI type
            pdb_id (str): the PDB ID of the solved structure
            domain_chain (str): the chain in the solved structure that shows to the domain
            domain_start (int): the residue index (resi) that denotes the start of the domain in the solved structure
            domain_end (int): the residue index (resi) that denotes the end of the domain in the solved structure
            motif_chain (str): the chain in the solved structure that shows to the motif
            motif_start (int): the residue index (resi) that denotes the start of the motif in the solved structure
            motif_end (int): the residue index (resi) that denotes the end of the motif in the solved structure
            download_directory (str, optional): Absolute path to the directory where the annotated pdb structure will be downloaded into
        """
        self.dmi_name = dmi_name
        self.pdb_id = pdb_id
        self.domain_chain = domain_chain
        self.domain_start = domain_start
        self.domain_end = domain_end
        self.min_domain_sequence = ''
        # create a dict to store the residue index and the residue of domain chain
        self.domain_resi_to_oneletter = {} # resi_to_oneletter has str as key (resi) because sometimes pdb structure has insertion code in their residue index
        self.motif_chain = motif_chain
        self.motif_start = motif_start
        self.motif_end = motif_end
        self.min_motif_sequence = ''
        # create a dict to store the residue index and the residue of motif chain
        self.motif_resi_to_oneletter = {} # resi_to_oneletter has str as key (resi) because sometimes pdb structure has insertion code in their residue index
        if download_directory is None:
            self.download_directory = f'/Volumes/imb-luckgr/projects/dmi_predictor/DMI_AF2_PRS/DMI_types/{self.dmi_name}/'
        else:
            self.download_directory = download_directory

    def download_pdb(self):
        """Download the pdb structure into the default directory or specified directory
        """
        # check if .pdb file of the pdb_id already exists, if not download and move it to the right directory
        if not os.path.exists(f'{self.download_directory}{self.pdb_id.upper()}.pdb'):
            print(f'Downloading {self.pdb_id}.pdb...')
            cmd.cd(self.download_directory)
            cmd.fetch(self.pdb_id, type='pdb')
            cmd.reinitialize()
            os.rename(f'{self.download_directory}{self.pdb_id}.pdb', f'{self.download_directory}{self.pdb_id.upper()}.pdb')
            print(f'Downloaded {self.pdb_id}.pdb to {self.download_directory}!')

    def read_in_sequence(self):
        """Use PyMol API to iterate through the sequence and its index in a chain in a pdb file and save it as a dictionary. Using the dictionary, save also the minimal domain and motif sequence as an attribute. Run download_pdb function if the pdb file is not found
        """
        self.download_pdb()
        cmd.load(f'{self.download_directory}{self.pdb_id}.pdb')
        # read in the domain sequence in the form of {resi:oneletter}
        cmd.iterate(f'{self.pdb_id} and chain {self.domain_chain} and n. CA','self.domain_resi_to_oneletter[resi]=oneletter',space={'self':self})
        # read in the motif sequence in the form of {resi:oneletter}
        cmd.iterate(f'{self.pdb_id} and chain {self.motif_chain} and n. CA','self.motif_resi_to_oneletter[resi]=oneletter',space={'self':self})
        # read in minimal domain seq, include backslash before residue index so that PyMol can handle negative residue index
        min_domain_sequence = []
        cmd.iterate(f'{self.pdb_id} and chain {self.domain_chain} and resi \{self.domain_start}-\{self.domain_end} and n. CA','min_domain_sequence.append(oneletter)',space={'min_domain_sequence':min_domain_sequence})
        self.min_domain_sequence = ''.join([oneletter for oneletter in min_domain_sequence])
        # read in minimal motif seq
        min_motif_sequence = []
        cmd.iterate(f'{self.pdb_id} and chain {self.motif_chain} and resi \{self.motif_start}-\{self.motif_end} and n. CA','min_motif_sequence.append(oneletter)',space={'min_motif_sequence':min_motif_sequence})
        self.min_motif_sequence = ''.join([oneletter for oneletter in min_motif_sequence])
        print(f'Read in motif sequence: {self.pdb_id,self.motif_start,self.motif_end,self.min_motif_sequence}')
        cmd.reinitialize()

    def extract_minimal_DMI(self):
        """Extract the minimal DMI from a solved structure and renumber its domain chain and motif chain indices to start from 1 and save it as a .pdb file with the suffix _min_DMI
        """
        if not os.path.exists(f'{self.download_directory}{self.pdb_id}_min_DMI.pdb'):
            cmd.load(f'{self.download_directory}{self.pdb_id}.pdb')
            cmd.create(f'{self.pdb_id}_min_DMI', f'{self.pdb_id} and chain {self.domain_chain} and resi \{self.domain_start}-\{self.domain_end} or {self.pdb_id} and chain {self.motif_chain} and resi \{self.motif_start}-\{self.motif_end}')
            # Want to alter the name of the domain chain as chain A and motif chain as B but sometimes it coincides with the chain id given by the author, so I first alter the chains into names that will likely not coincide with any author provided chain name, then alter those chain names back to A and B
            cmd.alter(f'{self.pdb_id}_min_DMI and chain {self.domain_chain} and resi \{self.domain_start}-\{self.domain_end}', 'chain="domain"')
            cmd.sort()
            cmd.alter(f'{self.pdb_id}_min_DMI and chain {self.motif_chain} and resi \{self.motif_start}-\{self.motif_end}', 'chain="motif"')
            cmd.sort()
            cmd.alter(f'{self.pdb_id}_min_DMI and chain domain', 'chain="A"')
            cmd.sort()
            cmd.alter(f'{self.pdb_id}_min_DMI and chain motif', 'chain="B"')
            cmd.sort()

            # # reindex the domain and motif chain so that the start from 1
            # cmd.alter(f'{self.pdb_id}_min_DMI and chain A', f'resi=(str(int(resi)-{int(self.domain_start) - 1}))')
            # cmd.sort()
            # cmd.alter(f'{self.pdb_id}_min_DMI and chain B', f'resi=(str(int(resi)-{int(self.motif_start) - 1}))')
            # cmd.sort()

            # # read in the minimal domain and motif sequence'
            # min_domain_sequence = {}
            # cmd.iterate(f'{self.pdb_id}_min_DMI and chain A and n. CA','min_domain_sequence[resi]=oneletter',space={'min_domain_sequence':min_domain_sequence})
            # self.min_domain_sequence = ''.join([oneletter for oneletter in min_domain_sequence.values()])
            # min_motif_sequence = {}
            # cmd.iterate(f'{self.pdb_id}_min_DMI and chain B and n. CA','min_motif_sequence[resi],oneletter))',space={'min_motif_sequence':min_motif_sequence})
            # self.min_motif_sequence = ''.join([oneletter for resi, oneletter in min_motif_sequence])

            # save the extracted minimal DMI
            cmd.save(f'{self.download_directory}{self.pdb_id}_min_DMI.pdb', f'({self.pdb_id}_min_DMI)')
            cmd.reinitialize()

def read_in_annotated_DMI_structure(DMI_annot_file):
    """Read in relevant information from the file that contains manually annotated minimal DMI region information in solved structures and create instances of DMI_structure

    Args:
        DMI_annot_file (str): the path to the DMI annotation file

    Returns:
        DMI_structure_dict (dict): A dict that contains dmi_name as key and DMI_structure instance as value
    """
    DMI_structure_dict = {}
    df = pd.read_csv(DMI_annot_file,sep='\t')
    df = df[df['for_AF2_benchmark'] == 1].copy()
    for i, r in df.iterrows():
        DMI_structure_inst = DMI_structure(r['dmi_type'],r['pdb_id'],r['chain_domain'],int(r['chain_domain_start']),int(r['chain_domain_end']),r['chain_motif'],int(r['chain_motif_start']),int(r['chain_motif_end']))
        DMI_structure_inst.read_in_sequence()
        DMI_structure_inst.extract_minimal_DMI()
        DMI_structure_dict[r['dmi_type']] = DMI_structure_inst
    
    return DMI_structure_dict

def calculate_peptide_RMSD(peptide_residue_indices):
    """Function only used after a session is open. First align the two structures on their domain, then calculate the backbone and all-atom RMSD of the template and predicted peptide. If only specific residues in the peptide should be compared for RMSD calculation, the argument peptide_residue_indices can be used to provide a list of residue indices (has to be same between the template and predicted models). Display the superimposed models by showing the chains in different colors and highlighting the peptides as sticks

    Args:
        peptide_residue_indices (list of int or str): a list of peptide residue indices on which RMSD should be calculated (useful for extended and mutated models)
    
    Returns:
        RMSD_domain (float): RMSD from domain alignment
        num_align_atoms_domain: Number of atom aligned between the template and predicted domain
        align_score_domain: Raw alignment score from aligning the template and predicted domain
        num_align_resi_domain: Number of residue aligned between the template and predicted domain
        RMSD_backbone_peptide (float): RMSD of backbone atoms of peptide after alignment on domain
        RMSD_all_atom_peptide (float): RMSD of all atoms of peptide after alignment on domain
    """
    # remove the hydrogen atoms as they are not used for RMSD calculation
    cmd.extract('h_atoms', 'hydrogens', source_state=1, target_state=1)
    cmd.delete('h_atoms')
    cmd.remove('hydrogen')

    # align the AF2_wk and real_wk on their domain chain
    domain_super_out = cmd.align(mobile='AF2_wk and chain A',target='real_wk and chain A',cycles=0,object='domain_aln', mobile_state=1, target_state=1)
    RMSD_domain = domain_super_out[0]
    num_align_atoms_domain = domain_super_out[1]
    align_score_domain = domain_super_out[5]
    num_align_resi_domain = domain_super_out[6]
    print(f'RMSD_domain: {RMSD_domain}')

    # calculate the RMSD on the peptides without making a fit first using the structures that are previously aligned on their domains
    peptide_residue_indices = '+\u005c'.join([str(resi) for resi in peptide_residue_indices]) # use the Unicode for backslash
    print(f'Calculating RMSD of peptide on residue indices: {peptide_residue_indices}')
    RMSD_backbone_peptide = cmd.rms_cur(mobile=f'AF2_wk and chain B and resi \{peptide_residue_indices} and bb.',target=f'real_wk and chain B and resi {peptide_residue_indices} and bb.',mobile_state=1,target_state=1,matchmaker=0,cycles=0,object='peptide_super_bb')
    RMSD_all_atom_peptide = cmd.rms_cur(mobile=f'AF2_wk and chain B and resi \{peptide_residue_indices}',target=f'real_wk and chain B and resi {peptide_residue_indices}',mobile_state=1,target_state=1,matchmaker=0,cycles=0,object='peptide_super_all_atoms')

    print(f'RMSD_backbone_peptide: {RMSD_backbone_peptide}')
    print(f'RMSD_all_atom_peptide: {RMSD_all_atom_peptide}')

    # display the superimposed peptide
    cmd.hide('all')
    cmd.show('cartoon', 'AF2_wk or real_wk')
    cmd.show('sticks', f'AF2_wk and chain B and resi \{peptide_residue_indices} or real_wk and chain B and resi \{peptide_residue_indices}')
    cmd.color('skyblue', '/real_wk/*/A')
    cmd.color('aquamarine', '/real_wk/*/B')
    cmd.color('tv_orange', '/AF2_wk/*/A')
    cmd.color('yelloworange', '/AF2_wk/*/B')
    cmd.color('atomic', f'/AF2_wk/*/B and not elem C')
    cmd.color('atomic', f'/real_wk/*/B and not elem C')
    cmd.orient(f'/real_wk/*/A')
    cmd.center(f'/real_wk/*/B')

    return RMSD_domain, num_align_atoms_domain,align_score_domain,num_align_resi_domain,RMSD_backbone_peptide,RMSD_all_atom_peptide

def process_calculate_model_RMSD(DMI_structure_inst,predicted_model):
    """Extract the minimal DMI region as previously annotated in template model from predicted model (mainly for extended DMI models but the function should also work for minimal DMI models). Then, renumber the motif residue index in the predicted model to be the same as that of template model. Create a PyMol object using the domain chain and the renumbered motif chain from predicted model for RMSD calculation. RMSD calculation is done by using the function calculate_peptide_RMSD. Additionally, extract the minimal domain chain and the minimal motif region from the renumbered motif chain and save it as _min.pdb file for DockQ calculation

    Args:
        DMI_structure_inst (DMI_structure): DMI_structure instance of the DMI type of the predicted model
        predicted_model (str): absolute path to the predicted model
        
    Returns:
        predicted_model_min.pdb: A pdb file containing the minimal DMI region and its residue index renumbered
    """
    predicted_model_name = os.path.splitext(os.path.split(predicted_model)[1])[0]
    # extract predicted motif sequence
    cmd.load(predicted_model)
    print(predicted_model)
    # load also the processed template model for reference
    cmd.load(f'{DMI_structure_inst.download_directory}{DMI_structure_inst.pdb_id}_min_DMI.pdb')
    predicted_motif_seq = []
    cmd.iterate(f'{predicted_model_name} and chain B and n. CA','predicted_motif_seq.append(oneletter)',space={'predicted_motif_seq':predicted_motif_seq})
    predicted_motif_seq = ''.join(predicted_motif_seq)
    print(predicted_motif_seq)

    # create a new object where the domain chain and the renumbered motif chain from the predicted model will be saved in
    cmd.create(f'AF2_wk', f'{predicted_model_name}')
    # create a new object from the processed template model to have a constant object name
    cmd.create(f'real_wk', f'{DMI_structure_inst.pdb_id}_min_DMI')

    # use regex matching to find where the minimal domain and motif sequence in solved structure match in the predicted domain and motif sequence
    match = re.finditer(f'(?=({DMI_structure_inst.min_motif_sequence}))',predicted_motif_seq)
    print(f'Minimal motif sequence in solved structure: {DMI_structure_inst.min_motif_sequence}')
    print(f'Motif sequence in predicted model: {predicted_motif_seq}')
    match_start = [m.start() + 1 for m in match]
    if len(match_start) == 1: # Only one match found, use the match start to renumber the model to be the same as the solved structure
        print(f'Regex matched at only one position: {match_start[0]}')
        cmd.alter(f'AF2_wk and chain B',f'resi=(int(resi)+{int(DMI_structure_inst.motif_start) - int(match_start[0])})')
        cmd.sort()
    else: # multiple match found, likely due to peptide sequence being too degenerate and the predicted model is a long extension
        # do the matching again by adding one residue N terminal to the minimal peptide
        print('Multiple matches found by matching minimal DMI to predicted sequence, trying N-terminal extension by 1 residue...')
        new_motif_start = DMI_structure_inst.motif_start - 1
        new_motif_seq = ''
        for i in range(new_motif_start,DMI_structure_inst.motif_end + 1):
            new_motif_seq += DMI_structure_inst.motif_resi_to_oneletter.get(str(i)) # resi_to_oneletter has str as key (resi)
        print(DMI_structure_inst.motif_resi_to_oneletter)
        match = re.finditer(f'(?=({new_motif_seq}))',predicted_motif_seq)
        match_start = [m.start() + 1 for m in match]
        print(f'Matching {new_motif_seq} to AF2 motif chain {predicted_motif_seq} returned {match_start}...')
        cmd.alter(f'AF2_wk and chain B',f'resi=(str(int(resi) + {int(DMI_structure_inst.motif_start) - 1 - int(match_start[0])}))')
        cmd.sort()

    # make a list of residue index for RMSD calculation
    peptide_residue_indices = []
    cmd.iterate(f'real_wk and chain B and resi \{DMI_structure_inst.motif_start}-\{DMI_structure_inst.motif_end} and n. CA','peptide_residue_indices.append(resi)',space={'peptide_residue_indices':peptide_residue_indices})

    # run calculate_peptide_RMSD function
    RMSD_domain, num_align_atoms_domain,align_score_domain,num_align_resi_domain,RMSD_backbone_peptide,RMSD_all_atom_peptide = calculate_peptide_RMSD(peptide_residue_indices)

    # retrieve the start and end of the aligned domain in the AF2_wk model
    AF2_aln_resi = []
    cmd.iterate('domain_aln and AF2_wk and n. CA','AF2_aln_resi.append(resi)',space={'AF2_aln_resi':AF2_aln_resi})

    # extract the domain chain and the minimal motif region from the renumbered motif chain to save it for DockQ calculation
    cmd.save(f'{predicted_model[:-4]}_min.pdb',f'AF2_wk and chain A and resi {AF2_aln_resi[0]}-{AF2_aln_resi[-1]} or AF2_wk and chain B and resi \{DMI_structure_inst.motif_start}-\{DMI_structure_inst.motif_end}')

    return RMSD_domain, num_align_atoms_domain,align_score_domain,num_align_resi_domain,RMSD_backbone_peptide,RMSD_all_atom_peptide

def calculate_known_motif_plddt(DMI_structure_inst):
        """Mainly used for extended models. Using the DMI_structure_inst, retrieve the region where known motif starts and ends in an annotated structure. Calculate the plddt of the known motif region
        
        Args:
            DMI_structure_inst (DMI_structure): An instance of previously annotated DMI structure

        Returns:
            known_motif_avg_plddt (float): calculated average plddt of known motif region, saved as attribute of self
        """
        known_motif_plddt = []
        cmd.iterate(f'AF2_wk and chain B and resi \{DMI_structure_inst.motif_start}-\{DMI_structure_inst.motif_end} and n. CA','known_motif_plddt.append(b)',space={'known_motif_plddt':known_motif_plddt})
        known_motif_avg_plddt = np.mean(known_motif_plddt)
        
        return known_motif_avg_plddt

def calculate_DockQ(predicted_model,template_model):
    """Calculate the DockQ metrics of a predicted model in reference to a provided template model by first running fix_numbering.pl to create an alignment file for the predicted model and use this alignment file to calculate the DockQ score in comparison to its template model

    Args:
        predicted_model (str): the absolute path to the processed predicted model (e.g. /Volumes/../ranked_0_min.pdb)
        template_model (str): the absolute path to the processed template model (e.g. /Volumes/../DEG_APCC_KENBOX_2/4GGD_min_DMI.pdb)

    Returns:
        DockQ_metrics (dict): DockQ metrics saved in a dict
    """
    # set up the path to DockQ and the fix_numbering.pl script
    current_path = os.path.abspath(__file__)
    one_level_up = os.path.dirname(current_path)
    DockQ_folder_path = f'{os.path.dirname(one_level_up)}/DockQ/'
    DockQ_path = os.path.join(DockQ_folder_path,'DockQ.py')
    fix_numbering_path = os.path.join(DockQ_folder_path,'scripts/fix_numbering.pl')

    # prepare a log file in the same folder as predicted model to store output of DockQ program
    prediction_folder, predicted_model_name = os.path.split(predicted_model)
    log_file = open(os.path.join(prediction_folder,'DockQ_log.log'),'a')
    log_file.write(f'Processing model {predicted_model_name}\n')

    try: # for some reason, minimal DMI requires the ranked_x.pdb file to be aligned using fix_numbering.pl first then use the ranked_x.pdb.fixed file for DockQ calculation
        # launch subprocess to first generate the .fixed file needed for DockQ calculation
        fix_numbering_process = [fix_numbering_path, predicted_model, template_model]
        result = subprocess.run(fix_numbering_process,capture_output=True,text=True,check=True)
        log_file.write(f'{result.stdout}\n')

        # launch subprocess to run DockQ script to compute DockQ score using the template model and predicted model with fixed numbering
        DockQ_process = ['python3', DockQ_path, f'{predicted_model}.fixed', template_model,'-short']
        result = subprocess.run(DockQ_process,capture_output=True,text=True,check=True)
        log_file.write(f'{result.stdout}\n')
        log_file.close()
    
    except: # for some reason, DockQ does not work on .fixed file, so I created this try and except to silence the error
        # launch subprocess to run DockQ script to compute DockQ score using the template model and predicted model with fixed numbering
        DockQ_process = ['python3', DockQ_path, f'{predicted_model}', template_model,'-short']
        result = subprocess.run(DockQ_process,capture_output=True,text=True,check=True)
        log_file.write(f'{result.stdout}\n')
        log_file.close()

    # parse subprocess output for relevant information
    result = result.stdout.split('\n')[-2]
    metrics = []
    values = []
    for i, ele in enumerate(result.split(' ')[:-3]):
        if i % 2 == 0:
            metrics.append(ele)
        else:
            values.append(ele)
    DockQ_metrics = {m:v for m, v in zip(metrics,values)}
    print(DockQ_metrics)

    # sleep the program for 1 second to avoid the error subprocess.CalledProcessError: died with <Signals.SIGTRAP: 5>.
    print('Sleeping for 1 second to avoid the error subprocess.CalledProcessError...')
    time.sleep(1)

    return DockQ_metrics

def process_calculate_mutated_model_RMSD(DMI_structure_inst,predicted_model):
    """Takes a predicted model done with mutated motif sequence and identify the residues that are not mutated using the name of the predicted model. Identify a list of residue index of the unmutated residue and run calculate_peptide_RMSD on these residues

    Args:
        DMI_structure_inst (DMI_structure): DMI_structure instance of the DMI type of the predicted model
        predicted_model (str): the absolute path to the processed predicted model (e.g. /Volumes/../run50_DEG_APCC_KENBOX_2_4GGD_SKENV.SGENV/ranked_0.pdb)

    Returns:
        peptide_residue_indices (list of int): a list of peptide residue indices on which RMSD should be calculated
    """
    # parse out the prediction_name
    prediction_folder, prediction_model = os.path.split(predicted_model)
    _, prediction_name = os.path.split(prediction_folder)

    predicted_model_name = os.path.splitext(os.path.split(predicted_model)[1])[0]
    # extract predicted motif sequence
    cmd.load(predicted_model)
    # load also the processed template model for reference
    cmd.load(f'{DMI_structure_inst.download_directory}{DMI_structure_inst.pdb_id}_min_DMI.pdb')

    # retrieve the predicted (mutated) sequence
    predicted_motif_seq = []
    cmd.iterate(f'{predicted_model_name} and chain B and n. CA','predicted_motif_seq.append(oneletter)',space={'predicted_motif_seq':predicted_motif_seq})
    predicted_motif_seq = ''.join(predicted_motif_seq)

    # retrieve the motif sequence from the template
    template_motif_seq = []
    cmd.iterate(f'{DMI_structure_inst.pdb_id}_min_DMI and chain B and n. CA','template_motif_seq.append(oneletter)',space={'template_motif_seq':template_motif_seq})
    template_motif_seq = ''.join(template_motif_seq)

    # create a new object where the domain chain and the renumbered motif chain from the predicted model will be saved in
    cmd.create(f'AF2_wk', f'{predicted_model_name}')
    # create a new object from the processed template model to have a constant object name
    cmd.create(f'real_wk', f'{DMI_structure_inst.pdb_id}_min_DMI')

    # renumber the template model so that it has the same index as the predicted model
    cmd.alter('real_wk and chain B',f'resi=(str(int(resi)-{int(DMI_structure_inst.motif_start) - 1}))')
    cmd.sort()

    # some solved motifs are < 4 residues long, while AF requires at least 4 residues for prediction. This step checks how many residues are solved and use it to slice the predicted sequence for the correct length. Important: This assumes that all extension to 4 residues are by one residue at the N terminus!
    if len(predicted_motif_seq) != len(template_motif_seq):
        print('Different length between solved motif and predicted motif sequence!')
        len_diff = len(predicted_motif_seq) - len(template_motif_seq)
        predicted_motif_seq = predicted_motif_seq[-len(template_motif_seq):]
        cmd.alter('AF2_wk and chain B',f'resi=(str(int(resi)-{len_diff}))')
        cmd.sort()

    # identify the unmutated residues between the template and predicted models
    unmutated_resi = []
    for i, res_pair in enumerate(zip(predicted_motif_seq,template_motif_seq)):
        if res_pair[0] == res_pair[1]:
            unmutated_resi.append(i+1)
    print(f"Predicted motif sequence: {predicted_motif_seq}")
    print(f"Solved motif sequence: {template_motif_seq}")
    print(f'Unmutated residues: {unmutated_resi}')

    # run calculate_peptide_RMSD function
    RMSD_domain, num_align_atoms_domain,align_score_domain,num_align_resi_domain,RMSD_backbone_peptide,RMSD_all_atom_peptide = calculate_peptide_RMSD(unmutated_resi)
    
    return RMSD_domain, num_align_atoms_domain,align_score_domain,num_align_resi_domain,RMSD_backbone_peptide,RMSD_all_atom_peptide

# if __name__ == '__main__':
#     # minimal or ext models
#     DMI_structure_dict = read_in_annotated_DMI_structure('/Volumes/imb-luckgr/projects/dmi_predictor/DMI_AF2_PRS/AF2_DMI_structure_PRS - Sheet1.tsv')
#     run_path = '/Users/chopyanlee/test_AF/run37'
#     for dir in os.listdir(run_path):
#         dir_path = os.path.join(run_path,dir)
#         if os.path.isdir(dir_path):
#             _, dmi_name = os.path.split(dir_path)
#             dmi_name = dmi_name.lstrip('run37_')
#             dmi_name = dmi_name[:-5]
#             # dmi_name, _ = dmi_name.split('.')
#             # dmi_name = '_'.join(dmi_name.split('_')[:4])
#             for i in range(5):
#                 print(f'{dmi_name},{dir_path}/ranked_{i}.pdb')
#                 DMI_structure_inst = DMI_structure_dict.get(dmi_name)
#                 # unmutated_resi = identify_unmutated_residue(DMI_structure_dict.get(dmi_name),f'{dir_path}/ranked_{i}.pdb')
#                 # print(unmutated_resi)
#                 # calculate_peptide_RMSD(peptide_residue_indices=unmutated_resi)
#                 process_calculate_model_RMSD(DMI_structure_inst,f'{dir_path}/ranked_{i}.pdb')
#                 calculate_peptide_RMSD()
#                 cmd.save(f'{dir_path}/ranked_{i}_super.pse')
#                 cmd.reinitialize()
#                 DockQ_metrics = calculate_DockQ(predicted_model=f'{dir_path}/ranked_{i}.pdb',template_model=f'{DMI_structure_inst.download_directory}{DMI_structure_inst.pdb_id}_min_DMI.pdb')
    # renumber_predicted_model()
    # calculate_peptide_RMSD()
    # calculate_DockQ()

    # mutated models
    # identify_unmutated_residue()
    # calculate_peptide_RMSD()

    # for mutated peptide calculation, use identify_unmutated_residue to open the session, use the DMI_structure_inst to load the model and the annotated motif start and end to renumber the predicted model
    # need to write different function to parse out dmi_name from different prediction type
    # matchmaker 0 better than 1