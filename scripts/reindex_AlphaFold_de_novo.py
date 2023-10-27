# This script reindexes AlphaFold predictions with a given residue index. Designed to take AlphaFold de novo predictions that has the naming convention of uniprot_idA_fragmentstartA_fragmentendA.uniprot_idB_fragmentstartB_fragmentendB
# Author: Chop Yan Lee

from pymol import cmd
import argparse, os

def reindex_model(pse_file,startA,startB):
    cmd.load(pse_file)
    obj_list = cmd.get_object_list('(all)')
    if 'renumbered' not in obj_list:
        cmd.create('renumbered','all and chain A or all and chain B')
        cmd.alter(f'/renumbered//A',f'resi=(str(int(resi)+{int(startA)-1}))')
        cmd.alter(f'/renumbered//B',f'resi=(str(int(resi)+{int(startB)-1}))')
        cmd.save(pse_file)
        print(f'Reindex prediction saved in {pse_file}!')
    else:
        print(f'Renumbered object already in {pse_file}, skipping this one!')
    cmd.reinitialize()

def parse_fragment_start(run_id,prediction_name):
    regionA, regionB = prediction_name.lstrip(f'{run_id}_').split('.')
    startA = int(regionA.split('_')[2])
    startB = int(regionB.split('_')[2])

    return startA, startB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-run_ids', type=str, help='Run IDs for metrics calculation', dest='run_ids')
    parser.add_argument('-path_to_run', type=str, help='Either provide a path to a folder where multiple runs of AlphaFold predictions are contained and specify the run_ids to be processed or use -path_to_prediction to specify a folder that you want to process, include "/" at the end', dest='path_to_run')
    args = parser.parse_args()
    run_ids = vars(args)['run_ids']
    path_to_run = vars(args)['path_to_run']

    for run_id in run_ids.split(','):
        print(f"Processing {path_to_run}run{run_id}/")
        for file in os.listdir(f'{path_to_run}run{run_id}/'):
            if os.path.splitext(file)[-1] == '.fasta':
                prediction_name = os.path.splitext(file)[0]
                startA, startB = parse_fragment_start(run_id=f'run{run_id}',prediction_name=prediction_name)
                print(startA,startB)
                if os.path.exists(f'{path_to_run}run{run_id}/{prediction_name}/ranking_debug.json'):
                    for i in range(5):
                        pse_file = f'{path_to_run}run{run_id}/{prediction_name}/ranked_{i}_contacts.pse'
                        reindex_model(pse_file,startA,startB)
        print(f"Completed the processing of {path_to_run}run{run_id}/!")

if __name__ == "__main__":
    main()

# python3 ~/AlphaFold_benchmark/scripts/reindex_AlphaFold_de_novo.py -run_ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 -path_to_run /Volumes/imb-luckgr/imb-luckgr2/projects/AlphaFold/DMIPred_NDD/