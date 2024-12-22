import torch.onnx
import sys
import os
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
sys.path.insert(1, '/app/Marabou')
from maraboupy import Marabou
from maraboupy import MarabouCore
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

options = Marabou.createOptions(verbosity = 0)
model_filename = "/app/Marabou/project/fmnist_dnn.onnx"
transform = transforms.Compose([
        transforms.ToTensor(),
    ])
training_data = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
test_data = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

train_dataloader=DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

network = Marabou.read_onnx(model_filename)
inputVars = network.inputVars[0][0]
outputVars = network.outputVars[0][0]

iterator = iter(train_dataloader)
inputs, classes = next(iterator)
image = inputs[0].numpy()[0]
correct_class = classes[0].item()

network_output = network.evaluateWithoutMarabou([image])[0]
predicted_class = np.argmax(network_output)
print("inputVars structure:", inputVars)
print("outputVars structure:", outputVars)
print("network.outputVars structure:", network.outputVars)
print("inputVars structure shape:", inputVars.shape)
print("outputVars structure shape:", outputVars.shape)
print("inputVars structure type:", type(inputVars))
print("outputVars structure type:", type(outputVars))

plt.title(f"correct:{correct_class}, predicted: {predicted_class}")
plt.imshow(image.reshape(28,28), cmap='gray')

epsilon = 0.01
for h in range(inputVars.shape[0]):
  for w in range(inputVars.shape[1]):
    network.setLowerBound(inputVars[h][w], image[h][w] - epsilon)
    network.setUpperBound(inputVars[h][w], image[h][w] + epsilon)

result={}
# sys.stdout = open('fmnist_dnn_result_0.01.txt', 'w')
start_time = time.time()
for i in range(outputVars.shape[0]):
  if i != correct_class:
    # add max constraint
    network.addMaxConstraint(set(outputVars), outputVars[i])
    # solve
    exit_code, vals, stats = network.solve(verbose = False, options = options)
    # if solution found, break
    print(exit_code)
    if len(vals) > 0:
      for j, var in enumerate(outputVars):
        print(f"output {j}: {vals[var]}")
      print(f"maxclass: {i}")
      result['result'] = exit_code
      result['counterexample'] = vals if exit_code == "sat" else None
      result['stats'] = stats
      break

result['result'] = 'unsat'
result['counterexample'] = None
result['stats'] = ""
end_time = time.time()
print("Robustness verification 결과:", result['result'])
print(f"Robustness verification 시간: {end_time - start_time:.4f}초")
# sys.stdout.close()



# def get_image_from_marabou(vals, inputVariables):
#   adversarial_image = [[] for _ in range(inputVariables.shape[1])]
#   for h in range(inputVariables.shape[0]):
#     for w in range(inputVariables.shape[1]):
#       adversarial_image[h].insert(w, vals[inputVariables[h][w]])
#   adversarial_image = np.array(adversarial_image)

#   return adversarial_image


# adversarial_image = get_image_from_marabou(vals, inputVars)
# network_output = network.evaluateWithoutMarabou([adversarial_image])[0]

# predicted_class = np.argmax(network_output)

# plt.title(f"correct:{correct_class}, predicted: {predicted_class}")
# plt.imshow(adversarial_image, cmap='gray')
# # plt.show()

