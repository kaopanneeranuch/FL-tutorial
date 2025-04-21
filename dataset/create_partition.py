import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
from pathlib import Path

# Define transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST training and test sets
trainset = datasets.MNIST(root="MNIST_data", train=True, download=True, transform=transform)
testset = datasets.MNIST(root="MNIST_data", train=False, download=True, transform=transform)

# Split training data into 3 equal parts
total_len = len(trainset)
split_len = total_len // 3
torch.manual_seed(42)
part1, part2, part3 = random_split(trainset, [split_len] * 3)

# Define exclusion logic
def exclude_digits(dataset, excluded_digits):
    indices = [i for i, (_, label) in enumerate(dataset) if label not in excluded_digits]
    return torch.utils.data.Subset(dataset, indices)

# Apply exclusion rules
part1 = exclude_digits(part1, excluded_digits=[1, 3, 7])
part2 = exclude_digits(part2, excluded_digits=[2, 5, 8])
part3 = exclude_digits(part3, excluded_digits=[4, 6, 9])

# Optional: include subsets of testset for analysis
def include_digits(dataset, included_digits):
    indices = [i for i, (_, label) in enumerate(dataset) if label in included_digits]
    return torch.utils.data.Subset(dataset, indices)

testset_137 = include_digits(testset, [1, 3, 7])
testset_258 = include_digits(testset, [2, 5, 8])
testset_469 = include_digits(testset, [4, 6, 9])

# Output directory
output_dir = Path("dataset")
output_dir.mkdir(exist_ok=True)

# Save partitions
torch.save(part1, output_dir / "part0.pt")
torch.save(part2, output_dir / "part1.pt")
torch.save(part3, output_dir / "part2.pt")

# Save shared test sets
torch.save(testset, output_dir / "testset.pt")
torch.save(testset_137, output_dir / "testset_137.pt")
torch.save(testset_258, output_dir / "testset_258.pt")
torch.save(testset_469, output_dir / "testset_469.pt")

print("Data partitions and test sets saved successfully.")
