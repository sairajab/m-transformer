import torch
import os
import h5py
from tqdm import tqdm
from Bio import SeqIO
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig
from biom import load_table
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from dataset_sparse import MicrobiomeSparseDataset
from dataset_sparse import collate_fn as sparse_collate_fn
import random
from embedding_loader import compute_and_save_embeddings
from optimized_embedding_loader import MemoryMappedEmbeddingLoader as EmbeddingLoader 
from itertools import product
from unifrac import unweighted
import torch.nn.functional as F
from biom.util import biom_open

from torch.utils.data import SubsetRandomSampler

from torch.utils.data import Sampler
import numpy as np
import random


class StratifiedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, n_bins=5):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.n_bins = n_bins

        # Create quantile bins
        self.binned_labels = pd.qcut(self.labels, q=n_bins, labels=False, duplicates='drop')
        self.indices = np.arange(len(labels))

        # Group sample indices by bin
        self.bins = {i: self.indices[self.binned_labels == i].tolist() for i in range(n_bins)}

        # Shuffle inside each bin (no fixed seed)
        for bin_indices in self.bins.values():
            random.shuffle(bin_indices)

        # Collect all indices while maintaining stratification
        self.all_indices = []
        bins_copy = {k: v.copy() for k, v in self.bins.items()}
        while any(bins_copy.values()):
            for bin_idx in range(n_bins):
                if bins_copy[bin_idx]:
                    self.all_indices.append(bins_copy[bin_idx].pop())

        # Pad the list to make it divisible by batch_size (optional but safe)
        remainder = len(self.all_indices) % self.batch_size
        if remainder > 0:
            pad = self.batch_size - remainder
            self.all_indices += self.all_indices[:pad]  # repeat from start

    def __iter__(self):
        return iter(self.all_indices)

    def __len__(self):
        return len(self.all_indices)



def one_body_out(sample_ids , donor_id = "D12"):
    
    test_ids = []
    train_samples = []

    
    for id in sample_ids:
        if donor_id in id:
            test_ids.append(id)
        else:
            train_samples.append(id) 
    
    #train_ids, val_ids = train_test_split(train_samples, test_size=0.1)
    
    return train_samples, test_ids
    
    


len_mers = 5
bases = ['A', 'C', 'G', 'T']
kmers = [''.join(p) for p in product(bases, repeat=len_mers)]
kmer_to_index = {kmer: idx+2 for idx, kmer in enumerate(kmers)}
base_to_index = {base: idx+2 for idx, base in enumerate(bases)}
CLS_IDX = 1
PAD_TOKEN_IDX = 0
def seq_to_kmer_indices(seq, k=5):
    return [kmer_to_index[seq[i:i+k]] for i in range(len(seq) - k + 1) if seq[i:i+k] in kmer_to_index]

def seq_to_base_indices(seq):
    return [base_to_index[base] for base in seq if base in base_to_index]

def pad_or_truncate(seq, max_len=288):
    if len(seq) >= max_len:
        return seq[:max_len]
    else:
        return torch.cat([seq, torch.full((max_len - len(seq),), PAD_TOKEN_IDX, dtype=torch.long)])

def one_hot_encode(sequence, max_len=288):
    """
    One-hot encode a nucleotide sequence.
    
    Args:
        sequence (str): Nucleotide sequence (A, C, G, T)
        max_len (int): Maximum length of the sequence
    
    Returns:
        torch.Tensor: One-hot encoded tensor of shape (max_len, 4)
    """
    one_hot = torch.zeros(max_len, 4, dtype=torch.float32)
    for i, base in enumerate(sequence):
        if i < max_len:
            if base == 'A':
                one_hot[i, 0] = 1.0
            elif base == 'C':
                one_hot[i, 1] = 1.0
            elif base == 'G':
                one_hot[i, 2] = 1.0
            elif base == 'T':
                one_hot[i, 3] = 1.0
    return one_hot



def tokenize_sequences(sequences, tokenizer, max_len=512):
    tokenized_seqs = {}
    seq = "TACAGAGGGTGCAAGCGTTGTTCGGAATCATTGGGCGTAAAGGGCGCGTAGGCGGTTTATCAAGTCGAATGTGAAAGCCCAGGGCTCAACCTTGGAAGTGCATCCGAAACTGGTAGACTAGAATCTCGGAGAGGGTGGTGGAATTCCCAGTGTAGAGGTGAAATTCGTAGATATTGGGAGGAACACCGGTGGCGAAGGCGACCACCTGGACAGAGATTGACGCTGAGGCGCGAGAGCGTGGGGAGCAAACAGG"

    encoded = tokenizer(seq, 
                        truncation=True,
                        return_tensors='pt')

    input_ids = encoded['input_ids'][0].tolist()

    print("Number of non-padding tokens:", sum(t != tokenizer.pad_token_id for t in input_ids))
    print("Length of input_ids:", len(input_ids))
    print("Tokenizer model max length:", tokenizer.model_max_length)


    
    for header, seq in sequences.items():
        # Tokenize sequence into k-mers and convert to token IDs
        # `return_tensors` omitted since we're storing token IDs
        tokenized = tokenizer(seq, padding='max_length', truncation=True, max_length=max_len)
        tokenized_seqs[header] = tokenized['input_ids']
        print(len(seq))
        print(seq)
        print(tokenizer.pad_token_id)  # Should print 3
        # Count tokens not equal to padding id (3)
        count_non_padding = sum(t != 3 for t in tokenized['input_ids'])
        print("Number of non-padding tokens:", count_non_padding)
        print(tokenized['input_ids'], len(tokenized['input_ids']))
        break

    return tokenized_seqs

def one_hot_encode(input_ids, vocab_size):
    # input_ids: [batch_size, seq_len]
    return F.one_hot(input_ids, num_classes=vocab_size).float()

def self_trained_mlm(embedding_path, sequences, device):
    
    from transformers import AutoTokenizer, AutoModel

    model_path = "asv_bert_mlm_250/checkpoint-41820"  # e.g., "outputs/checkpoint-27880"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model = model.to(device)
    model.eval()  # Disable dropout etc.
    compute_and_save_embeddings(
        sequences=sequences,
        tokenizer=tokenizer,
        model=model,
        output_path=embedding_path,
        device=device,
        max_length=250  # Adjust as needed
    )

    
def dna_bert(embedding_path, sequences, device):
    
            config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
            tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
            dnabert_model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)
            dnabert_model = dnabert_model.to(device)

            
            compute_and_save_embeddings(
                    sequences=sequences,
                    tokenizer=tokenizer,
                    model=dnabert_model,
                    output_path=embedding_path,
                    device=device
            )
            
            # Free up memory
            #del dnabert_model
            torch.cuda.empty_cache()

def finetuned_dna_bert(embedding_path, sequences, device):
            checkpoint_path = "dnabert-finetuned-16s-no-pad/checkpoint-37170"  # adjust this if full path is needed

            tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
            finetuned_model = AutoModel.from_pretrained(checkpoint_path,trust_remote_code=True)
            dnabert_model = finetuned_model.to(device)
            compute_and_save_embeddings(
                sequences=sequences,
                tokenizer=tokenizer,
                model=dnabert_model,
                output_path=embedding_path,
                device=device
            )
            
            # Free up memory
            #del dnabert_model
            torch.cuda.empty_cache()



class DataProcessor:
    def __init__(self, args):
        """
        Class to load and process microbiome/sequence data.
        
        Args:
            biom_file (str): Path to BIOM file with abundances
            sequence_file (str): Path to FASTA file (or IDs only)
            target_file (str): Path to sample targets file
            embedding_dir (str): Directory to store embeddings
            unifrac_tree (str, optional): Path to UniFrac tree
            train_encoder (bool): Whether to process for encoder training
            kmer_embedding (bool): Whether to process with k-mer embeddings
        """
        self.config = args


        # Loaded data
        self.table = None
        self.sequences = {}
        self.sample_targets = None
        self.distances = None
        self.unifrac_tree = None
        self.seqs_processed = None
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.indoor_samples = None
        self.outdoor_samples = None

    def load_biom(self):
        table = load_table(self.config.biom_file)
        self.table = table
        return table

    def load_sequences(self):
        """
        Here we’re just filling sequences with observation IDs.
        Replace with SeqIO.parse(self.sequence_file, "fasta") if needed.
        """
        if self.table is None:
            raise ValueError("BIOM table not loaded yet. Call load_biom() first.")

        s_ids = self.table.ids(axis="observation")
        self.sequences = {s_id: s_id for s_id in s_ids}
        return self.sequences

    def compute_unifrac(self):
        if self.unifrac_tree is None:
            return None

        rand = np.random.random(1)[0]
        temp_path = f"/tmp/temp{rand}.biom"
        with biom_open(temp_path, "w") as f:
            self.table.to_hdf5(f, "aam")
        self.distances = unweighted(temp_path, self.config.tree_path)
        os.remove(temp_path)
        return self.distances

    def load_targets(self):
        ids = self.table.ids()
        targets_df = pd.read_csv(self.config.metadata_file, delimiter="\t")
        
        targets_df["sample_name"] = targets_df["sample_name"].astype(str)
        filtered = targets_df[targets_df["sample_name"].isin(ids)]
        filtered = filtered[filtered["dataset_type"] == "train"].copy()
        print("Number of samples in targets after filtering:", len(filtered))
        self.sample_targets = dict(
            zip(filtered["sample_name"], filtered["add_0c"] / 100)
        )
        return self.sample_targets
    
    def load_targets_multitask(self):
        ids = self.table.ids()
        targets_df = pd.read_csv(self.config.metadata_file, delimiter="\t")
        ## for some reasons sheds table has 13810 at the beginning of the sample ids
        ## concat all sample ids with 13810 for sheds data only
        #targets_df["SampleID"] = "13810." + targets_df["SampleID"].astype(str)
        filtered = targets_df[targets_df["sample_name"].isin(ids)]
        filtered = filtered[filtered["dataset_type"] == "train"].copy()
        print("Number of samples in targets after filtering:", len(filtered))
        ## create bool for indoor/outdoor        
        indoor_samples = filtered[filtered["env"] == "indoor"]["sample_name"].tolist()
        outdoor_samples = filtered[filtered["env"] != "indoor"]["sample_name"].tolist()

        print("Number of indoor samples:", len(indoor_samples))
        print("Number of outdoor samples:", len(outdoor_samples))

        self.sample_targets = {}
        self.sample_targets = dict(
            zip(filtered["sample_name"], filtered["add_0c"].astype(float) / 100)
        )
        self.indoor_samples = indoor_samples
        self.outdoor_samples = outdoor_samples
        return self.sample_targets
    
    def process_sequences(self):
        """
        Process sequences into embeddings/k-mers/encoder inputs depending on mode.
        """
        if self.config.kmer_embeddings:
            seqs_kmers = {}
            max_len = 289
            for header, seq in self.sequences.items():
                indices = [CLS_IDX] + seq_to_kmer_indices(seq)
                seqs_kmers[header] = pad_or_truncate(
                    torch.tensor(indices, dtype=torch.long), max_len=max_len
                )
            return seqs_kmers

        elif self.config.train_encoder:
            seqs_kmers = {}
            max_len = 150
            for header, seq in self.sequences.items():
                indices = seq_to_base_indices(seq)
                seqs_kmers[header] = pad_or_truncate(
                    torch.tensor(indices, dtype=torch.long), max_len=max_len
                )
            return seqs_kmers

        else:

            if not os.path.exists(self.config.embedding_file):
                print("Computing embeddings...")
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                finetuned_dna_bert(self.config.embedding_file, self.sequences, device)
            return self.config.embedding_file
    def match_samples(self):
        if self.table is None or self.sample_targets is None:
            raise ValueError("Table or sample targets not loaded.")

        sample_ids = set(self.table.ids())
        target_ids = set(self.sample_targets.keys())
        common_ids = sample_ids.intersection(target_ids)

        if not common_ids:
            raise ValueError("No matching sample IDs between table and targets.")

        # Filter table and targets to only include common IDs
        self.table = self.table.filter(common_ids, axis="sample", inplace=False)
        self.sample_targets = {k: v for k, v in self.sample_targets.items() if k in common_ids}

        print(f"Number of samples after matching: {len(common_ids)}")
        return self.table, self.sample_targets

    def run(self, multitask=False):
        """Full pipeline."""
        self.load_biom()
        self.load_sequences()
        if self.config.tree_path:
            self.compute_unifrac()
        if multitask:
            self.load_targets_multitask()
        else:
            self.load_targets()
        self.match_samples()
        self.seqs_processed = self.process_sequences()
    
    def load_data(self, multitask=False):

        self.run(multitask=multitask)
        sample_ids = list(self.sample_targets.keys())

        train_samples, test_samples = one_body_out(sample_ids, donor_id = self.config.heldout)
        # Create train/val target dictionaries
        train_targets = {sample_id: self.sample_targets[sample_id] for sample_id in train_samples}
        test_targets = {sample_id: self.sample_targets[sample_id] for sample_id in test_samples}
        self.train_data = (train_samples, train_targets)
        self.test_data = (test_samples, test_targets)
    
        
    def sample_data(self, epoch):
    # Create datasets

        train_samples, train_targets = self.train_data
        
        if self.val_data is None and epoch ==0:
            train_samples, val_samples = train_test_split(train_samples, test_size=0.1 , random_state=42)
            train_targets = {sample_id: self.sample_targets[sample_id] for sample_id in train_samples}
            val_targets = {sample_id: self.sample_targets[sample_id] for sample_id in val_samples}
            self.val_data = (val_samples, val_targets)
            self.train_data = (train_samples, train_targets)
            print("Validation data created with", len(val_samples), "samples.")            

        print(f"Number of training samples: {len(train_samples)}")
        print(f"Number of validation samples: {len(self.val_data[0])}")
        print(f"Number of test samples: {len(self.test_data[0])}")
        
        train_table = self.table.filter(train_samples , inplace = False)
        val_table = self.table.filter(val_samples , inplace = False)

        if not self.config.kmer_embeddings and self.config.embedding_file is None:
                train_dataset = MicrobiomeSparseDataset(
                    biom_table=train_table,
                    one_hot_seqs=self.seqs_processed,
                    sample_targets=train_targets, 
                    random_vec=False, 
                    seed = epoch 
                )
                val_dataset = MicrobiomeSparseDataset(
                    biom_table=val_table,
                    one_hot_seqs=self.seqs_processed,
                    sample_targets=val_targets, 
                    random_vec=False,
                    seed = epoch 
                )
        elif not self.config.kmer_embeddings and self.config.embedding_file is not None:
            # Use the embedding path to create the dataset
            embedding_loader = EmbeddingLoader(self.config.embedding_file)
            train_dataset = MicrobiomeSparseDataset(
                    biom_table=train_table,
                    sample_targets=train_targets,
                    embedding_loader=embedding_loader,
                    random_vec=False,
                    seed = epoch,
                    env = (self.indoor_samples, self.outdoor_samples) 
                )
                
            val_dataset = MicrobiomeSparseDataset(
                        biom_table=val_table,
                        sample_targets=val_targets,
                        embedding_loader=embedding_loader,
                        random_vec=False,
                        seed = epoch ,
                        env = (self.indoor_samples, self.outdoor_samples)
                    )
        elif self.config.kmer_embeddings and self.config.embedding_file is None:
            # Use the kmer_seqs to create the dataset
            train_dataset = MicrobiomeSparseDataset(
                    biom_table=train_table,
                    kmer_seqs=self.seqs_processed,
                    sample_targets=train_targets, random_vec=False
                )
            val_dataset = MicrobiomeSparseDataset(
                        biom_table=val_table,
                        kmer_seqs=self.seqs_processed,
                        sample_targets=val_targets, random_vec=False
                    )
        
        train_y = train_dataset.get_targets()
        val_y = val_dataset.get_targets()
        train_targets = [train_y[k] for k in train_dataset.sample_ids]
        val_targets = [val_y[k] for k in val_dataset.sample_ids]

        return train_dataset, val_dataset

    def sample_test_data(self, random_vector=False):

        test_samples, test_targets = self.test_data
        test_table = self.table.filter(test_samples, inplace=False)


        if not self.config.kmer_embeddings and self.config.embedding_file is None:
                test_dataset = MicrobiomeSparseDataset(
                    biom_table=test_table,
                    one_hot_seqs=self.seqs_processed,
                    sample_targets=test_targets, 
                    random_vec=random_vector, 
                    seed = None
                )
        elif not self.config.kmer_embeddings and self.config.embedding_file is not None:
            # Use the embedding path to create the dataset
            embedding_loader = EmbeddingLoader(self.config.embedding_file)
            test_dataset = MicrobiomeSparseDataset(
                    biom_table=test_table,
                    sample_targets=test_targets,
                    embedding_loader=embedding_loader,
                    random_vec=random_vector,
                    seed = None,
                    env = (self.indoor_samples, self.outdoor_samples)
                )
        elif self.config.kmer_embeddings and self.config.embedding_file is None:
            # Use the kmer_seqs to create the dataset
            test_dataset = MicrobiomeSparseDataset(
                    biom_table=test_table,
                    kmer_seqs=self.seqs_processed,
                    sample_targets=test_targets, random_vec=random_vector
                )
        else:
            raise ValueError("Unknown configuration for test data sampling.")


        return test_dataset




class Arguments:
    def __init__(self, biom_file, metadata_file, embedding_file, tree_path = None,  heldout = "D12", embedding="DNABERT", normalize=False):
        
        self.normalize = normalize
        self.heldout = heldout
        self.embedding = embedding
        self.biom_file = biom_file
        self.metadata_file = metadata_file
        self.embedding_file = embedding_file
        self.kmer_embeddings  = False
        self.train_encoder = False
        self.tree_path = tree_path
        if self.embedding == "kmers":
            self.kmer_embeddings = True
        elif self.embedding == "train_encoder":
            self.train_encoder = True
            
        
def shuffle_indices(samples_dict):
    items = list(samples_dict.items())  
    random.shuffle(items)  
    return dict(items)


     


