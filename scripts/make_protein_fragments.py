# This script contains classes that store information of proteins in order to define their ordered and disordered regions. The definition of ordered and disordered regions within a protein is intended to facilitate the fragmentation of disordered regions. The defined regions are then paired with similarly defined regions of its interacting protein to generate fasta files that can be submitted to AF pipeline for de novo interaction interface prediction.
# # Author: Chop Yan Lee
import pandas as pd
import numpy as np
from pymol import cmd
import os, sys, itertools, igraph, json

protein_info_directory = '/Users/chopyanlee/Coding/Python/DMI/protein_sequences_and_features/'
alphafold_directory = '/Volumes/imb-luckgr/imb-luckgr2/projects/AlphaFold/'
pymol_colors = cmd.get_color_indices()[2:] # [('white',0),('black',1),...], skip white and black due to background color 
pymol_colors = [(color,i) for color, i in pymol_colors if color != 'green'] # skip green as it is the default color

def parse_pae_file(pae_json_file):
    """Parse PAE file produced by AF
    
    Args:
    pae_json_file (str): path to the PAE file
    
    Returns:
    An n residue by n residue matrix
    
    Acknowledgement:
    https://github.com/tristanic/pae_to_domains"""

    with open(pae_json_file, 'rt') as f:
        data = json.load(f)[0]
    
    r1, d = data['residue1'],data['distance']
    size = max(r1)
    matrix = np.empty((size,size))
    matrix.ravel()[:] = d

    return matrix

def domains_from_pae_matrix_igraph(pae_matrix, pae_power=1, pae_cutoff=5, graph_resolution=1):
    '''
    Takes a predicted aligned error (PAE) matrix representing the predicted error in distances between each 
    pair of residues in a model, and uses a graph-based community clustering algorithm to partition the model
    into approximately rigid groups.

    Arguments:

        * pae_matrix: a (n_residues x n_residues) numpy array. Diagonal elements should be set to some non-zero
          value to avoid divide-by-zero warnings
        * pae_power (optional, default=1): each edge in the graph will be weighted proportional to (1/pae**pae_power)
        * pae_cutoff (optional, default=5): graph edges will only be created for residue pairs with pae<pae_cutoff
        * graph_resolution (optional, default=1): regulates how aggressively the clustering algorithm is. Smaller values
          lead to larger clusters. Value should be larger than zero, and values larger than 5 are unlikely to be useful.

    Returns: a series of lists, where each list contains the indices of residues belonging to one cluster.

    Acknowledgement:
    https://github.com/tristanic/pae_to_domains
    '''
    weights = 1/pae_matrix**pae_power

    g = igraph.Graph()
    size = weights.shape[0]
    g.add_vertices(range(size))
    edges = np.argwhere(pae_matrix < pae_cutoff)
    sel_weights = weights[edges.T[0], edges.T[1]]
    g.add_edges(edges)
    g.es['weight']=sel_weights

    vc = g.community_leiden(weights='weight', resolution_parameter=graph_resolution/100, n_iterations=-1)
    membership = np.array(vc.membership)
    from collections import defaultdict
    clusters = defaultdict(list)
    for i, c in enumerate(membership):
        clusters[c].append(i)
    clusters = list(sorted(clusters.values(), key=lambda l:(len(l)), reverse=True))

    return clusters

class Protein:
    """Class to store protein information"""
    def __init__(self,uniprot_id: str):
        """Make an instance of a protein.
        
        Args:
        UniProt ID (str) of the protein
        """
        self.uniprot_id = uniprot_id
        self.sequence = None
        self.plddt = []
        # PAE of the protein saved in the form of n res by n res matrix
        self.info_dir = f'{alphafold_directory}protein_info/{self.uniprot_id}/'
        if os.path.exists(f'{self.info_dir}AF-{self.uniprot_id}-F1-model_v2.pdb'):
            self.AF_structure = f'{self.info_dir}AF-{self.uniprot_id}-F1-model_v2.pdb'
        elif os.path.exists(f'{self.info_dir}AF-{self.uniprot_id}-F1-model_v4.pdb'):
            self.AF_structure = f'{self.info_dir}AF-{self.uniprot_id}-F1-model_v4.pdb'
        else:
            print(f'Unable to find AlphaFold model in the folder {self.info_dir}')
        self.AF_sequence = None
        # self.PAE = parse_pae_file(f'{self.info_dir}AF-{self.uniprot_id}-F1-predicted_aligned_error_v2.json')
        self.iupred = []
        # store instances of regions in a list
        self.regions = []
        # store fragmentated disordered region
        self.disordered_fragments = []

    def extract_sequence_from_AF(self):
        """Load the self.AF_structure file of AF predicted structure and extract the sequence used by AF for prediction
        
        Args:
        self (Protein): instance of Protein to assess its self.AF_structure attribute
        
        Returns:
        A string of sequence of the predicted monomeric structure of a protein"""
        model_name = os.path.split(self.AF_structure)[1]
        model_name = os.path.splitext(model_name)[0]
        cmd.load(self.AF_structure)
        seq = []
        cmd.iterate(f"{model_name} and n. CA","seq.append(oneletter)",space={'seq':seq})
        cmd.reinitialize()
        self.AF_sequence = ''.join(seq)

    def read_in_sequence_iupred(self):
        """Read in the sequence and IUPred score of the protein"""
        seq_path = os.path.join(protein_info_directory,'human_protein_sequences')
        iupred_path = os.path.join(protein_info_directory,'human_protein_sequences_features/IUPred_long')
        if os.path.exists(os.path.join(seq_path,f'{self.uniprot_id}.txt')):
            with open(os.path.join(seq_path,f'{self.uniprot_id}.txt'), 'r') as f:
                lines = [line.strip() for line in f.readlines()]
                self.sequence = lines[1]
            with open(os.path.join(iupred_path,f'{self.uniprot_id}_iupredlong.txt'), 'r') as f:
                lines = [line.strip() for line in f.readlines()]
                for line in lines[1:]:
                    _,_, iupred = line.split('\t')
                    self.iupred.append(float(iupred))
        else:
            print(f'{self.uniprot_id} sequence not found!')
            sys.exit()

    def check_equal_sequence_length(self):
        """As the my look-up files for IUPred were downloaded two years ago, check to ensure that the length of IUPred is the same as the number of residue in the AF predicted structure, otherwise one has to redownload
        
        Returns:
        True if equal, else False"""
        self.extract_sequence_from_AF()
        if self.sequence == self.AF_sequence:
            print(f'Look up file of {self.uniprot_id} has the same sequence as that used by AlphaFold for prediction. All good!')
        else:
            print(f'Look up file of {self.uniprot_id} DO NOT have the same sequence as that used by AlphaFold for prediction. The IUPred calculated using the look up file might not be accurate in this case!')

    def define_regions(self,graph_resolution=0.5,determine_type=False):
        """Makes use of iupred, PAE and plddt of the sequence to determine ordered and disordered regions in the protein.

        Args:
        graph_resolution (float): parameter given to the clustering algorithm
        determine_type (bool): whether or not to automatically detect the type (ordered or disordered) of a region

        Returns:
        List of region instances"""
        clusters = domains_from_pae_matrix_igraph(self.PAE,graph_resolution=graph_resolution)
        for cluster in clusters:
            start = min(cluster)
            end = max(cluster)
            region = Region(start,end,'ordered')
            region.calculate_avg_iupred_PAE(iupred=self.iupred)
            if determine_type:
                region.determine_type()
            self.regions.append(region)

        print(f'{len(clusters)} ordered regions found in {self.uniprot_id}!')

    def define_disordered_regions(self):
        """To be executed after checking and refining the ordered regions detected from PAE matrix. Creates region instances of regions that are outside of ordered or transmembrane ones as disordered regions.
        
        Returns:
        List of region instances"""
        sorted_ordered_regions = sorted([region for region in self.regions if (region.type == 'ordered') or (region.type == 'transmembrane') or (region.type == 'extracellular')],key=lambda x: x.start)
        index = -1

        # if the whole protein is disordered, make the whole protein as a disordered region
        if len(sorted_ordered_regions) == 0:
            region = Region(1,len(self.sequence),'disordered')
            region.calculate_avg_iupred_PAE(iupred=self.iupred)
            self.regions.append(region)
            len(self.sequence)
        else: # if exists and more than 5 residues, make trailing regions at N terminus a disordered region
            if sorted_ordered_regions[0].start >= 5:
                start = 0
                end = sorted_ordered_regions[0].start - 1
                region = Region(start,end,'disordered')
                region.calculate_avg_iupred_PAE(iupred=self.iupred)
                self.regions.append(region)
            # same with the trailing regions at the C terminus
            if len(self.sequence) - sorted_ordered_regions[-1].end >= 5:
                start = sorted_ordered_regions[-1].end + 1
                end = len(self.sequence) - 1
                region = Region(start,end,'disordered')
                region.calculate_avg_iupred_PAE(iupred=self.iupred)
                self.regions.append(region)
            # also make the regions in between two ordered regions as disordered
            for _ in range(len(sorted_ordered_regions) - 1):
                index += 1
                if sorted_ordered_regions[index+1].start - sorted_ordered_regions[index].end >= 5:
                    start = sorted_ordered_regions[index].end + 1
                    end = sorted_ordered_regions[index+1].start -1
                    region = Region(start,end,'disordered')
                    region.calculate_avg_iupred_PAE(iupred=self.iupred)
                    self.regions.append(region)

    def fragmentate_disordered_regions(self,fragment_sizes):
        """Checks if any region is of disordered type and make fragments of it by fragmenting the disordered region in sizes specified with the argument fragment_size if the disordered region length is > 50. The fragments are to be paired with minimal domains from its interaction partner.

        Args:
        fragments_sizes (list of int): Specify the size of the disordered fragments to fragment from the disordered regions e.g. [10,20,30] was used in AlphaFold paper.

        Returns:
        List of region instances that store the fragments of disordered regions"""
        disord_regions = [region for region in self.regions if region.type == 'disordered']
        for region in disord_regions:
            added_fragments = set() # set that checks if a fragment's start and end has been appended to self.disordered_fragments
            number_frag = 0
            # if length is longer than 50, make fragments by sliding and merging fragments into longer fragments
            if region.length >= 50:
                slide_steps = [0,1] # not to slide or slide
                for frag_size, slide_step in itertools.product(fragment_sizes,slide_steps):
                    slide_length = int(frag_size / 2)
                    for i in range(region.length // frag_size):
                        if i == 0:
                            start = 0
                        else:
                            start = i * frag_size
                        end = (i + 1) * frag_size
                        frag_start = region.start + start + (slide_step * slide_length)
                        frag_end = region.start + end + (slide_step * slide_length)
                        frag_end = frag_end if frag_end <= region.end else region.end # if sliding goes beyond sequence, stop
                        if (frag_start, frag_end) in added_fragments:
                            continue
                        added_fragments.add((frag_start,frag_end))
                        frag = Region(frag_start,frag_end,'disordered')
                        frag.calculate_avg_iupred_PAE(iupred=self.iupred)
                        self.disordered_fragments.append(frag)
                        number_frag += 1
            # if length is shorter than 50 and greater than 10, make fragments by sliding without merging
            elif (region.length < 50) & (region.length > 10):
                number_frag_to_make = 5
                window_size = int(region.length / 2)
                slide_length = (region.length - window_size) // number_frag_to_make
                for i in range(number_frag_to_make + 1): # +1 because fragmenting by cutting the region in half is the first fragment
                    frag_start = region.start + i * slide_length
                    frag_end = region.start + window_size + i * slide_length
                    if (frag_start, frag_end) in added_fragments:
                            continue
                    added_fragments.add((frag_start,frag_end))
                    frag = Region(frag_start,frag_end,'disordered')
                    frag.calculate_avg_iupred_PAE(iupred=self.iupred)
                    self.disordered_fragments.append(frag)
                    number_frag += 1
                    # after sliding, the trailing fragment(s) with >=5 residues should be made into a fragment too
                    if frag_start - region.start + 1 >= 5:
                        if (region.start, frag_start) in added_fragments:
                            continue
                        added_fragments.add((region.start, frag_start))
                        frag = Region(region.start,frag_start,'disordered')
                        frag.calculate_avg_iupred_PAE(iupred=self.iupred)
                        self.disordered_fragments.append(frag)
                        number_frag += 1
                    if region.end - frag_end +1 >= 5:
                        if (frag_end, region.end) in added_fragments:
                            continue
                        added_fragments.add((frag_end, region.end))
                        frag = Region(frag_end,region.end,'disordered')
                        frag.calculate_avg_iupred_PAE(iupred=self.iupred)
                        self.disordered_fragments.append(frag)
                        number_frag += 1
            # disordered region less than 10, too short to fragmentate
            #  as all disordered regions will be used for pairing too, skip appending this region to self.disordered_fragments
            else:
                continue
            print(f'Fragmented the disordered regions of spanning from {region.start+1} to {region.end+1} into {number_frag} fragments!')

    def write_out_regions(self,include_fragments=False,out_name=None):
        """Write out the ordered regions detected from PAE matrix. This is intended to facillitate the manual checking and further refinement of ordered regions detected from PAE matrix.

        Args:
        incliude_fragments (boolean): include fragmentated disordered regions
        out_name (str): specify name of the output file
        
        Returns:
        .tsv file containing the information of all the ordered regions detected from PAE matrix"""
        columns = ['start','end','type','length','avg_iupred','color_on_structure','approved']
        df = pd.DataFrame(columns=columns)
        for color_index, region in zip(pymol_colors,self.regions):
            # + 1 to start and end because residue starts from 1
            if region.type == 'disordered':
                color = None
            else:
                color = color_index[0]
            df.loc[len(df)] = [region.start+1,region.end+1,region.type,region.length,region.avg_iupred,color,region.approved]
        if include_fragments:
            for frag in self.disordered_fragments:
                df.loc[len(df)] = [frag.start+1,frag.end+1,frag.type,frag.length,frag.avg_iupred,None,None]
        
        out_name = 'annotated_regions.tsv' if out_name is None else out_name
        df.round(3).to_csv(f'{self.info_dir}{out_name}',sep='\t')
        print(f'Regions and/or fragments are saved in {self.info_dir}{out_name}!')

    def _read_in_refined_regions(self):
        """To be used with make_annotated_pdb. After manually checking the ordered regions detected from PAE matrix, use the regions_from_PAE_checked.tsv file to make a list of refined regions and replace all the regions previously detected from PAE matrix.
        
        Returns:
        A new list of refined regions that replace the region list before"""
        self.regions = []
        df = pd.read_csv(f'{self.info_dir}annotated_ordered_disordered_regions.tsv',sep='\t')
        for i, r in df.iterrows():
            start = r['start'] - 1
            end = r['end'] - 1
            region = Region(start,end,r['type'])
            region.calculate_avg_iupred_PAE(iupred=self.iupred)
            region.approved = r['approved']
            self.regions.append(region)

    def make_annotated_pdb(self,checked=False):
        """Create a pymol session file (.pse) for an AF structure with its ordered regions colored in different colors. This is intended to facillitate manual checking and further refinement of ordered regions detected from PAE matrix.
        
        Args:
        checked (bool): Once regions detected from PAE matrix has been checked and saved in regions_from_PAE_checked.tsv, toggle this argument to True so that the regions can be reannotated and saved in a pymol session file
        
        Returns:
        .pse file containing the AF structure annotated with its structured regions"""
        pdb_path = self.AF_structure
        cmd.load(pdb_path)

        # if ordered regions already checked before and regions_from_PAE_checked.tsv exists, read in the refined ordered regions and annotate them on the structure
        if checked:
            self._read_in_refined_regions()
            save_path = f'{pdb_path[:-4]}_annotated_checked.pse'
        # else, simply use the regions detected from PAE matrix to annotate the structure
        else:
            save_path = f'{pdb_path[:-4]}_annotated.pse'
        for color_index, region in zip(pymol_colors,self.regions):
            color, index = color_index
            # + 1 to start and end because residue starts from 1
            cmd.color(color=color,selection=f"resi {region.start + 1}-{region.end + 1}")
        cmd.save(save_path)
        cmd.reinitialize()

    def __repr__(self):
        """Represent protein instance as its sequence, ordered and disordered regions"""
        print(f'UniProt ID: {self.uniprot_id}\nSequence: {self.sequence}\n')
        items = []
        # + 1 to start and end because residue starts from 1
        if any(self.regions):
            for region in self.regions:
                approved = "Approved" if region.approved == 1 else "Unchecked"
                items.append(f'Region of interest: {region.start+1}-{region.end+1}, length {region.length}, average IUPred {region.avg_iupred:.2f}, {region.type}, {approved}')
        if any(self.disordered_fragments):
            for fragment in self.disordered_fragments:
                items.append(f'Disordered fragment: {fragment.start+1}-{fragment.end+1}, length {fragment.length}, average IUPred {fragment.avg_iupred:.2f}, {fragment.type}')
        return '\n'.join(items)

class ProteinPair:
    """Class that takes a pair of protein instances and generates output for AlphaFold prediction"""
    def __init__(self,proteinA: Protein,proteinB: Protein):
        """Make an instance of a protein pair using protein instance. Internally sort the protein instances given as argument according to their uniprot id in alphabetical order.

        Args:
        Protein instance of protein A: Protein
        Protein instance of protein B: Protein
        """
        self.proteinA = sorted([proteinA,proteinB],key=lambda x: x.uniprot_id)[0]
        self.proteinB = sorted([proteinA,proteinB],key=lambda x: x.uniprot_id)[1]
        self.intx_id = '_'.join([self.proteinA.uniprot_id,self.proteinB.uniprot_id])
        # list of paired fragments stored in tuple
        self.paired_fragments = []

    def pair_fragments(self):
        """Make combinatorial pairing of ordered and disordered regions and fragments of proteinA and proteinB, and ordered and ordered regions of proteinA and proteinB.

        Returns:
        Paired fragments stored in the form of tuple of two region instances, i.e. (proteinA.region, proteinB.region), with proteinA always being the first. The tuple is then appended to self.paired_fragments"""
        ordered_regionA = [region for region in self.proteinA.regions if region.type == 'ordered']
        ordered_regionB = [region for region in self.proteinB.regions if region.type == 'ordered']
        disordered_regionA = self.proteinA.disordered_fragments + [region for region in self.proteinA.regions if region.type == 'disordered']
        disordered_regionB = self.proteinB.disordered_fragments + [region for region in self.proteinB.regions if region.type == 'disordered']

        # ordered + ordered pairing to test potential domain-domain interface
        for paired_frag in itertools.product(ordered_regionA,ordered_regionB):
            self.paired_fragments.append(paired_frag)

        # ordered + disordered pairing to test potential domain-motif interface
        # orderedA + disorderedB
        for paired_frag in itertools.product(ordered_regionA,disordered_regionB):
            self.paired_fragments.append(paired_frag)
        # disorderedA + orderedB
        for paired_frag in itertools.product(disordered_regionA,ordered_regionB):
            self.paired_fragments.append(paired_frag)

    def write_out_fragment_fasta_description(self,path_to_run, run_id):
        """Write out a fasta file for every paired fragment of the protein pair and a .tsv description file that contains information of each paired fragment.

        Args:
        path_to_run (str): Absolute path to the run folder
        run_id (str): The run ID folder in which the fasta files and .tsv description file are to be saved, e.g. 'run50'

        Returns:
        .fasta file containing the paired regions of proteinA and proteinB with name format protA_O_50_300.protB_D_12_20.fasta, indicating that ordered region from 50 to 300 from protein A is paired with disordered region from 12 to 20 from protein B.
        .tsv file containing the description of each pairing"""
        run_directory = os.path.join(path_to_run,run_id) + '/'

        # make the run directory if it does not exist
        if not os.path.exists(run_directory):
            os.makedirs(run_directory)

        # make a dataframe to store the description of paired fragments and generate the .fasta file simultaneously
        columns = ['intx_id','prediction_name','proteinA','region_typeA','region_startA','region_endA','region_lengthA','proteinB','region_typeB','region_startB','region_endB','region_lengthB']
        df = pd.DataFrame(columns=columns)
        for paired_frag in self.paired_fragments:
            fragA, fragB = paired_frag
            fragA_type_short = 'D' if fragA.type == 'disordered' else 'O'
            fragB_type_short = 'D' if fragB.type == 'disordered' else 'O'
            prediction_name = f'{run_id}_{self.proteinA.uniprot_id}_{fragA_type_short}_{fragA.start+1}_{fragA.end+1}.{self.proteinB.uniprot_id}_{fragB_type_short}_{fragB.start+1}_{fragB.end+1}'

            with open(f'{run_directory}{prediction_name}.fasta', 'w') as f:
                f.write(f'>{self.proteinA.uniprot_id}_{fragA_type_short}_{fragA.start+1}_{fragA.end+1}\n{self.proteinA.sequence[fragA.start:fragA.end+1]}\n>{self.proteinB.uniprot_id}_{fragB_type_short}_{fragB.start+1}_{fragB.end+1}\n{self.proteinB.sequence[fragB.start:fragB.end+1]}\n')

            df.loc[len(df)] = [self.intx_id,prediction_name,self.proteinA.uniprot_id,fragA.type,fragA.start+1,fragA.end+1,fragA.length,self.proteinB.uniprot_id,fragB.type,fragB.start+1,fragB.end+1,fragB.length]

        df['total_length'] = df['region_lengthA'] + df['region_lengthB']
        df.to_csv(f'{run_directory}paired_fragments_descriptions.tsv',sep='\t')
        print(f'{len(self.paired_fragments)} fasta files have been generated for all paired fragments!')
        print(f'Description file of paired fragments has been saved as {run_directory}paired_fragments_descriptions.tsv!')

    def __repr__(self):
        """Represent protein pair instance as its uniprot_idA and uniprot_idB, and their paired regions."""
        print(f'Protein A: {self.proteinA.uniprot_id}, Protein B: {self.proteinB.uniprot_id}, intx_id: {self.intx_id}\n')
        print(f'Number of paired fragments: {len(self.paired_fragments)}')
        for paired_frag in self.paired_fragments:
            fragA, fragB = paired_frag
            print(f'Protein A: {fragA.start}-{fragA.end}, {fragA.type} + Protein B: {fragB.start}-{fragB.end}, {fragB.type}')

class Region:
    """Class to store the information of a region of interest (ordered/disordered) in a protein"""
    def __init__(self,start,end,type):
        """Make an instance of region of a protein.

        Args:
        start (int): Start of the region of interest. Index from 0.
        end (int): End of the region of interest. Index from 0.
        type (str): Region of interest being ordered, disordered or transmembrane (excluded from pairing)
        avg_iupred (float)= Average of IUPredLong of the region
        avg_PAE (float)= Average of plddt of the region"""
        self.start = start
        self.end = end
        self.type = type
        self.length = end - start + 1
        self.avg_iupred = None
        self.approved = 0

    def calculate_avg_iupred_PAE(self,iupred):
        """Calculates the average of IUPred of the region.

        Args:
        iupred (list): a list of iupred score from a protein instance
        
        Returns:
        Modifies the instance variable self.avg_iupred"""
        self.avg_iupred = np.mean(iupred[self.start:self.end])

    def determine_type(self):
        """Use a series of criteria (avg_iupred,avg_PAE and length) to determine if a region is disordered
        
        Returns:
        Modifes the instance variable self.type to disordered if the instance meets the criteria"""
        if self.length <= 30 or self.avg_iupred >= 0.5 or self.avg_PAE >= 10:
            self.type = 'disordered'
        else:
            self.type = 'ordered'

if __name__ == "__main__":
    pass