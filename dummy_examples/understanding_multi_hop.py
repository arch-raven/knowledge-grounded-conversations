import torch
from torch_scatter import scatter_max

gamma = 1.0

bsz = 1
L = 2

# 5->7 earth->rock

concepts = [
    "eruption",
    "science",
    "pour",
    "volcano",
    "thing",
    "earth",
    "water",
    "rock",
    "lava",
    "substance",
]
distances = [0, 0, 0, 0, 0, 1, 1, 1, 2, 2]
source_concepts = ["eruption", "science", "pour", "volcano"]
target_concepts = ["lava", "substance"]
heads = [
    "eruption",
    "science",
    "pour",
    "volcano",
    "thing",
    "earth",
    "rock",
    "rock",
    "rock",
]
tails = [
    "lava",
    "earth",
    "water",
    "lava",
    "substance",
    "rock",
    "lava",
    "substance",
    "water",
]

mem = len(concepts)
mem_t = len(heads)
assert len(heads) == len(tails)

head = torch.tensor([[concepts.index(node) for node in heads]]).view(bsz, mem_t)
tail = torch.tensor([[concepts.index(node) for node in tails]]).view(bsz, mem_t)
triple_label = torch.tensor([[int(node in target_concepts) for node in tails]]).view(
    bsz, mem_t
)
distance = torch.tensor([distances]).view(bsz, mem)  # no padding
concept_label = torch.tensor(
    [[int(node in target_concepts) for node in concepts]]
).view(
    bsz, mem
)  # no padding

triple_prob = torch.rand(bsz, L, mem_t).view(bsz, L, mem_t)
assert mem >= head.max() and mem >= tail.max()


concept_probs = []
cpt_size = (triple_prob.size(0), triple_prob.size(1), distance.size(1))
init_mask = (
    torch.zeros_like(distance)
    .unsqueeze(1)
    .expand(*cpt_size)
    .to(distance.device)
    .float()
)
init_mask.masked_fill_((distance == 0).unsqueeze(1), 1)
final_mask = init_mask.clone()

init_mask.masked_fill_((concept_label == -1).unsqueeze(1), 0)
concept_probs.append(init_mask)

head = head.unsqueeze(1).expand(triple_prob.size(0), triple_prob.size(1), -1)
tail = tail.unsqueeze(1).expand(triple_prob.size(0), triple_prob.size(1), -1)
for step in range(3):
    """
    Calculate triple head score
    """
    # (BS, L, mem)
    node_score = concept_probs[-1]
    # head.shape (BS, mem_t) -> (BS, L, mem_t)
    triple_head_score = node_score.gather(2, head)
    triple_head_score.masked_fill_((triple_label == -1).unsqueeze(1), 0)
    """
    Method: 
        - avg:
            s(v) = Avg_{u \in N(v)} gamma * s(u) + R(u->v) 
        - max: 
            s(v) = max_{u \in N(v)} gamma * s(u) + R(u->v)
    """
    update_value = triple_head_score * gamma + triple_prob
    out = torch.zeros_like(node_score).to(node_score.device).float()
    if True:
        scatter_max(update_value, tail, dim=-1, out=out)
    else:
        scatter_mean(update_value, tail, dim=-1, out=out)
    out.masked_fill_((concept_label == -1).unsqueeze(1), 0)

    concept_probs.append(out)

"""
Natural decay of concept that is multi-hop away from source
"""
total_concept_prob = final_mask * -1e5
for prob in concept_probs[1:]:
    total_concept_prob += prob
# bsz x L x mem
