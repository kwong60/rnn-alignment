import matplotlib.pyplot as plt
import torch

history = torch.load('checkpoints/training_history.pth')

# weights from epoch 0
weights_epoch0 = history['weight_history'][0]['weights']['fc_h2ah_weight']

# weights from last epoch
weights_last = history['weight_history'][-1]['weights']['fc_h2ah_weight']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im1 = axes[0].imshow(weights_epoch0.numpy(), cmap='RdBu', aspect='auto')
axes[0].set_title('Recurrent Weights - Epoch 0')
axes[0].set_xlabel('From unit')
axes[0].set_ylabel('To unit')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(weights_last.numpy(), cmap='RdBu', aspect='auto')
axes[1].set_title(f'Recurrent Weights - Epoch {history["weight_history"][-1]["epoch"]}')
axes[1].set_xlabel('From unit')
axes[1].set_ylabel('To unit')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig('plots/weight_comparison.png', dpi=150)
plt.show()

print("Saved weight visualization!")