from dataset import Dataset

# Load the dataset
dataset = Dataset("data/indy_20160420_01.mat")

# Print statistics
print(dataset)
print()

# Test getting a trial
print("Testing get_trial()...")
trial = dataset.get_trial(3)
print(trial)
print()

# Test visualization methods
print("Testing visualization methods...")
dataset.visualize_waveforms(channel=1, output_file="test_waveforms.png")
dataset.visualize_trial_position(trial_number=3,
                                 output_file="test_trial_position.png")
dataset.visualize_trial_neural(trial_number=3,
                               output_file="test_trial_neural.png")

print("\nAll tests completed successfully!")
