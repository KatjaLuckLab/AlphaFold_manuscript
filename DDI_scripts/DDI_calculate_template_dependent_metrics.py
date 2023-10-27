# This script contains generic functions that calculates metrics from predicted models with the use of a template for domain-domain interfaces.
# Author: Chop Yan Lee
from pymol import cmd
import os, subprocess, time, sys
from renumber import renumber
sys.setrecursionlimit(10**5) # for renumber.py

def download_pdb(ddi_name,pdb_id):
    """Download the pdb structure of a DDI type to a default directory
    
    Args:
        ddi_name (str): DDI name
        pdb_id (str): PDB ID that shows the DDI
    """
    download_directory = f'/Volumes/imb-luckgr/projects/ddi_predictor/DDI_manual_curation/{ddi_name}/'
    pdb_id = pdb_id.upper()
    if not os.path.exists(f'{download_directory}'):
        os.makedirs(f'{download_directory}')
    # check if .pdb file of the pdb_id already exists, if not download and move it to the right directory
    if not os.path.exists(f'{download_directory}{pdb_id}.pdb'):
        print(f'Downloading {pdb_id}.pdb...')
        cmd.cd(download_directory)
        cmd.fetch(pdb_id, type='pdb')
        cmd.reinitialize()
        os.rename(f'{download_directory}{pdb_id}.pdb', f'{download_directory}{pdb_id}.pdb')
        print(f'Downloaded {pdb_id}.pdb to {download_directory}!')

def extract_minimal_DDI(ddi_name,pdb_id,domain_chainA,domain_startA,domain_endA,domain_chainB,domain_startB,domain_endB):
    """Extract the minimal DDI from a solved structure and save it as a .pdb file with the suffix _min_DMI
    
    Args:
        ddi_name (str): the name of DDI type
        pdb_id (str): the PDB ID of the solved structure
        domain_chainA (str): the chain in the solved structure that shows to the first domain of the DDI type
        domain_startA (int): the residue index (resi) that denotes the start of the domain1 in the solved structure
        domain_endA (int): the residue index (resi) that denotes the end of the domain1 in the solved structure
        domain_chainB (str): the chain in the solved structure that shows to the second domain of the DDI type
        domain_startB (int): the residue index (resi) that denotes the start of the domain2 in the solved structure
        domain_endB (int): the residue index (resi) that denotes the end of the domain2 in the solved structure
    """
    pdb_id = pdb_id.upper()
    ddi_directory = f'/Volumes/imb-luckgr/projects/ddi_predictor/DDI_manual_curation/{ddi_name}/'
    if not os.path.exists(f'{ddi_directory}{pdb_id}_min_DDI.pdb'):
        cmd.load(f'{ddi_directory}{pdb_id}.pdb')
        cmd.create(f'{pdb_id}_min_DDI', f'{pdb_id} and chain {domain_chainA} and resi \{domain_startA}-\{domain_endA} or {pdb_id} and chain {domain_chainB} and resi \{domain_startB}-\{domain_endB}')
        # Want to alter the name of the domain chain as chain A and motif chain as B but sometimes it coincides with the chain id given by the author, so I first alter the chains into names that will likely not coincide with any author provided chain name, then alter those chain names back to A and B
        cmd.alter(f'{pdb_id}_min_DDI and chain {domain_chainA}', 'chain="domainA"')
        cmd.sort()
        cmd.alter(f'{pdb_id}_min_DDI and chain {domain_chainB}', 'chain="domainB"')
        cmd.sort()
        cmd.alter(f'{pdb_id}_min_DDI and chain domainA', 'chain="A"')
        cmd.sort()
        cmd.alter(f'{pdb_id}_min_DDI and chain domainB', 'chain="B"')
        cmd.sort()

        # save the extracted minimal DMI
        cmd.save(f'{ddi_directory}{pdb_id}_min_DDI.pdb', f'({pdb_id}_min_DDI)')
        cmd.reinitialize()

def perform_RMSD_calculation(predicted_model,template_model):
    """Calculate the RMSD of smaller domain in the DDI in reference to the same domain in the solved structure by anchoring the predicted model and template model on their bigger domain
    
    Args:
        predicted_model (str): the absolute path to the processed predicted model (e.g. /Volumes/../ranked_0.pdb)
        template_model (str): the absolute path to the processed template model (e.g. /Volumes/../DDI_manual_curation/PF00023_PF07686/4NIK_min_DDI.pdb)
    
    Returns:
        RMSD_big_domain (float): RMSD from alignment on bigger domain
        RMSD_all_atom_small_domain (float): RMSD of all atoms of smaller domain after alignment on bigger domain
        RMSD_backbone_small_domain (float): RMSD of backbone atoms of smaller domain after alignment on bigger domain
        num_align_atoms_big_domain (int): Number of atom aligned between the template and predicted bigger domain
        align_score_big_domain (int): Raw alignment score from aligning the template and predicted bigger domain
        num_align_resi_big_domain (int): Number of residue aligned between the template and predicted bigger domain
    """
    save_path = os.path.splitext(predicted_model)[0]

    # load the 'real' structure and the predicted structure into pymol
    cmd.load(predicted_model, 'AF')
    cmd.load(template_model, 'real')
    cmd.extract('h_atoms', 'hydrogens', source_state=1, target_state=1)
    cmd.delete('h_atoms')
    cmd.remove('hydrogen')

    # make copies of the template and predicted model so that we work only with these copies
    cmd.create('AF_wk', 'AF')
    cmd.create('real_wk', 'real')

    # check the number of residues in each chain to decide which one is bigger for alignment
    chain_length_dict = {}
    cmd.iterate("AF_wk and n. CA","chain_length_dict[chain]=chain_length_dict.get(chain,0)+1",space={"chain_length_dict":chain_length_dict})
    print(chain_length_dict)
    # by default use chain A for alignment
    align_chain = 'A'
    RMSD_chain = 'B'
    # if chain A is bigger, then align on this chain
    if chain_length_dict.get('B') >= chain_length_dict.get('A'):
        align_chain = 'B'
        RMSD_chain = 'A'
    # align mobile=/AF_wk/*/A, target=/real_wk/*/A, cycles=0,mobile_state=1,target_state=1
    # as rms_cur for RMSD calculation later relies on residue index for RMSD calculation, I will reset the residue index of the smaller domain
    renumber(selection=f'real_wk and chain {RMSD_chain}')
    print(f'align chain = chain {align_chain}, RMSD_chain = chain {RMSD_chain}')

    # alignment on bigger chain
    domain_super_out = cmd.align(mobile=f"AF_wk and chain {align_chain}", target=f"real_wk and chain {align_chain}", cycles=0, object='big_domain_aln', mobile_state=1, target_state=1)
    RMSD_big_domain = domain_super_out[0]
    num_align_atoms_big_domain = domain_super_out[1]
    align_score_big_domain = domain_super_out[5]
    num_align_resi_big_domain = domain_super_out[6]
    print(f'RMSD domain = {RMSD_big_domain}')

    # RMSD calculation on smaller chain
    RMSD_backbone_small_domain = cmd.rms_cur(mobile=f"AF_wk and chain {RMSD_chain} and bb.", target=f"real_wk and chain {RMSD_chain} and bb.", mobile_state=1,target_state=1,matchmaker=4, cycles=0, object='small_domain_super_bb')
    RMSD_all_atom_small_domain = cmd.rms_cur(mobile=f'AF_wk and chain {RMSD_chain}', target=f'real_wk and chain {RMSD_chain}', mobile_state=1, target_state=1,matchmaker=4, cycles=0, object='small_domain_super_all_atoms')
    print(f'RMSD_backbone_small_domain: {RMSD_backbone_small_domain}')
    print(f'RMSD_all_atom_small_domain: {RMSD_all_atom_small_domain}')

    # display the superimposed models
    cmd.hide('all')
    cmd.show('cartoon', 'AF_wk or real_wk')
    cmd.color('skyblue', f'/real_wk/*/{align_chain}')
    cmd.color('aquamarine', f'/real_wk/*/{RMSD_chain}')
    cmd.color('tv_orange', f'/AF_wk/*/{align_chain}')
    cmd.color('yelloworange', f'/AF_wk/*/{RMSD_chain}')
    cmd.orient('AF_wk or real_wk')
    cmd.center('AF_wk or real_wk')
    cmd.save('/Users/chopyanlee/Coding/Python/DDI/AF_code_test/test.pse')

    return RMSD_big_domain, RMSD_all_atom_small_domain, RMSD_backbone_small_domain,num_align_atoms_big_domain, align_score_big_domain, num_align_resi_big_domain

def calculate_DockQ(predicted_model,template_model):
    """Calculate the DockQ metrics of a predicted model in reference to a provided template model by first running fix_numbering.pl to create an alignment file for the predicted model and use this alignment file to calculate the DockQ score in comparison to its template model

    Args:
        predicted_model (str): the absolute path to the processed predicted model (e.g. /Volumes/../ranked_0_min.pdb)
        template_model (str): the absolute path to the processed template model (e.g. /Volumes/../DDI_manual_curation/PF00023_PF07686/4NIK_min_DDI.pdb)

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