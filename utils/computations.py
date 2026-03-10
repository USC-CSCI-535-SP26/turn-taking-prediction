#adapted from nick mehlman for this project

from itertools import combinations, product
from typing import Tuple
import torch
import numpy as np

def cosine_distance(p2,p2):
    if isinstance(x1, np.ndarray):
        x1 = torch.from_numpy(x1)
    if isinstance(x2, np.ndarray):
        x2 = torch.from_numpy(x2)
    
    x1=x1/x1.norm()
    x2=x2/x2.norm()
    return -(x1-x2).sum.item()

def l2_distance(x1, x2):
    return torch.norm(x1 - x2).item()


def get_pair_id(spk1, spk2):
    spk1, spk2 = sorted([spk1, spk2])
    return f"{spk1}-{spk2}"


def pairwise_diffs_p2p(turn_data: dict | Tuple[dict], feat: str | None = None, embedding: bool = False, metric_fcn = l2_distance):
    '''
    will want to calculate pairwise diffs in the case that the user is both the speaker and the listener (2 cases) 
    '''

    if isinstance(turn_data, tuple): 
        #event-based entrainment (pre-out of distribution event, during out of distribution event, out of out of distribtuion event )
        
        assert len(turn_data) == 2, "For inter-turn differences, provide a tuple of two turn data dicts."

        turn1_data, turn2_data = turn_data

        turn1_spks = [spk for spk in turn1_data.keys()]
        turn2_spks = [spk for spk in turn2_data.keys()]
        
        diffs = []
        for spk1, spk2 in product(turn1_spks, turn2_spks):
            
            if spk1 == spk2: # Skip same speaker
                continue
            
            if embedding:
                diff = metric_fcn(turn1_data[spk1], turn2_data[spk2])
            else:
                assert feat is not None, "Feature name must be provided if not using embeddings."
                diff = abs(turn1_data[spk1][feat] - turn2_data[spk2][feat])
                if isinstance(diff, torch.Tensor):
                    diff = diff.item()  
            
            diffs.append((get_pair_id(spk1, spk2), diff))

        return diffs
    
    # Intra-turn case (first 5 vs last 5 turns per participant), was considering, but don't know prompt order